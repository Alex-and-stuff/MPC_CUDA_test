/*
* Run importCUDAc.m in E:\Research_Backup_NAS\碩一上\stuff\Matlab simulations\bicycle MPC simulation
* to plot the results of the below MPC results.
* be sure to store "costroll.csv", "predstate.csv" and "testFile.csv" to the matlab folder
*/

#include <iostream>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <curand_kernel.h>
#include <curand.h>
#include <chrono>

using namespace std;


#define SIZE		256
#define SHMEM_SIZE	256 
#define DSIZE		102400
#define USIZE		40

#define CONTROLSIZE 2				// Number of controls u = [vd,wd]
#define STATESIZE   4				// Number of states   x = [x,y,phi,v0]
#define T_HRZ       5.0f			// Prediction time hoeizon
#define F_STP		20.0f			// Frequency of prediction (control freq)
#define T_STP		1/F_STP			// Period of prediction
#define N_HRZ       T_HRZ*F_STP		// N steps in a time horizon
#define K			1024            // K-rollout predictions
#define V_MAX       2.5f			// Velocity command upper-bound
#define V_MIN       1.8f			// Velocity command lower-bound
#define W_MAX		1.0f			// Angular acceleration command upper-bound
#define W_MIN		-1.0f			// Angular acceleration command lower-bound
#define ROAD_WIDTH  2.0f			// Width of road (unit: m)
#define LAMBDA      100.0f
#define RAND_SCALAR 1.0f
#define RHO         0.3f

#define FG_RAD      10.0f
#define GH_RAD		10.0f
#define HI_RAD      10.0f
#define IF_RAD      10.0f

#define OFF_ROAD_COST	500.0f		// Penalty for leaving the road
#define COLLISION_COST	800.0f		// Penalty for colliding into an obstacle
#define TRACK_ERR_COST	10.0f		// Penalty for tracking error (parting from the middle)

#define ITERATIONS 4500

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0);

#define CUDA_SAFE_CALL(call) do { \
	cudaError_t err = call; \
	if (cudaSuccess != err) { \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.", \
			__FILE__, __LINE__, cudaGetErrorString(err)); \
		exit(EXIT_FAILURE); \
	} \
} while (0);

struct Control {
	float vd, wd;
};
struct State {
	float x, y, phi, v0;
};
struct Track {
	// Track fcn in the form of by + ax + c =0
	float b, a, c;
};
struct Pos {
	float x, y;
};

struct Obs {
	float x, y, r;
};




__global__ void sum_reduction(float* v, float* v_r) {
	/*  Perform sum reduction to an array of floats (here since the for
		loop iterates with a division of 2, the blockDim (#of threads)
		must be 2 to the nth power) !!Important!! */

	__shared__ float partial_sum[64];

	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
		//printf("result2: %.3f\n", partial_sum[0]);
		//printf("%4d::%.3f\n", blockIdx.x, v_r[blockIdx.x]);
	}
}


__host__ __device__ State bicycleModel(State state, Control u) {
	/*  Linear bicycle model, which geneates state_dot. Then by integrating the state_dot,
		we can then get the nest state x_k+1*/

	State state_dot;
	State state_next;
	float Ka = 2.54;
	state_dot.x = state.v0 * cosf(state.phi);
	state_dot.y = state.v0 * sinf(state.phi);
	state_dot.phi = u.wd;
	state_dot.v0 = Ka * state.v0 * (u.vd - state.v0);
	state_next.x = state.x + state_dot.x * T_STP;
	state_next.y = state.y + state_dot.y * T_STP;
	state_next.phi = state.phi + state_dot.phi * T_STP;
	state_next.v0 = state.v0 + state_dot.v0 * T_STP;
	return state_next;
}

__device__ void clampingFcn(Control* u_in) {
	/*	Clamping fcn for the perturbated command, acting as a
		hard contrain*/

	if (u_in->vd > V_MAX) { u_in->vd = V_MAX; }
	else if (u_in->vd < V_MIN) { u_in->vd = V_MIN; }
	if (u_in->wd > W_MAX) { u_in->wd = W_MAX; }
	else if (u_in->wd < W_MIN) { u_in->wd = W_MIN; }
}

__device__ float distanceFromTrack_old(float inner_f, float inner_g, float inner_h,
	float inner_i, Track* inner_fcn) {
	/*  Calculates the distance from the middle of a designated path
		(Here, from the Tetragon imposing the road infront of the schools library)*/

	float slope[] = { -inner_fcn[0].a, -inner_fcn[1].a, -inner_fcn[2].a, -inner_fcn[3].a };
	float distance = 0.0f;
	if (inner_f < 0 && inner_g < 0 && inner_h < 0 && inner_i > 0) {
		distance = fabs(fabs(inner_f) - 0.5 * ROAD_WIDTH / cosf(atanf(slope[0])));
	}
	else if (inner_f < 0 && inner_g > 0 && inner_h < 0 && inner_i > 0) {
		distance = fabs(sqrtf(pow(inner_f, 2) + pow(inner_g, 2)) - 0.5 * ROAD_WIDTH);
	}
	else if (inner_f > 0 && inner_g > 0 && inner_h < 0 && inner_i > 0) {
		distance = fabs(fabs(inner_g) - 0.5 * ROAD_WIDTH / cosf(atanf(slope[1])));
	}
	else if (inner_f > 0 && inner_g > 0 && inner_h > 0 && inner_i > 0) {
		distance = fabs(sqrtf(pow(inner_g, 2) + pow(inner_h, 2)) - 0.5 * ROAD_WIDTH);
	}
	else if (inner_f > 0 && inner_g < 0 && inner_h>0 && inner_i > 0) {
		distance = fabs(fabs(inner_h) - 0.5 * ROAD_WIDTH / cosf(atanf(slope[2])));
	}
	else if (inner_f > 0 && inner_g < 0 && inner_h>0 && inner_i < 0) {
		distance = fabs(sqrtf(pow(inner_h, 2) + pow(inner_i, 2)) - 0.5 * ROAD_WIDTH);
	}
	else if (inner_f > 0 && inner_g < 0 && inner_h < 0 && inner_i < 0) {
		distance = fabs(fabs(inner_i) - 0.5 * ROAD_WIDTH / cosf(atanf(slope[3])));
	}
	else if (inner_f > 0 && inner_g < 0 && inner_h < 0 && inner_i < 0) {
		distance = fabs(sqrtf(pow(inner_i, 2) + pow(inner_f, 2)) - 0.5 * ROAD_WIDTH);
	}
	else if (inner_f == 0 || inner_g == 0 || inner_h == 0 || inner_i == 0) {
		float inner_min = 0.0f;
		(fabs(inner_f) > fabs(inner_g)) ? inner_min = fabs(inner_g) : inner_min = fabs(inner_f);
		(inner_min > fabs(inner_h)) ? inner_min = fabs(inner_h) : inner_min = inner_min;
		(inner_min > fabs(inner_i)) ? inner_min = fabs(inner_i) : inner_min = inner_min;
		distance = inner_min - 0.5 * ROAD_WIDTH;
	}
	if (distance > ROAD_WIDTH / 2) {
		distance = 2; // Important to set it to 2 instead of 0 to ensure the vehicle tracks the designated path
	}

	return distance;
}

__device__ float pointToLineDist(float inner, Track line) {
	float a = line.a;
	float b = line.b;
	return fabs(inner) / sqrtf(a * a + b * b);
}

__device__ float distanceFromTrack(float* inner, float* nominal, Track* inner_fcn, Pos* turn_circle_intersec, State* state) {
	/*  Calculates the distance from the middle of a designated path
		(Here, from the Tetragon imposing the road infront of the schools library)*/

	float slope[] = {-inner_fcn[0].a, -inner_fcn[1].a, -inner_fcn[2].a, -inner_fcn[3].a};

	// Distance from the curves 
	// @TODO
	if (nominal[0] >= 0 && nominal[1] <= 0) {  // curve fg
		return fabs(sqrtf((state->x - turn_circle_intersec[0].x) * (state->x - turn_circle_intersec[0].x)
			+ (state->y - turn_circle_intersec[0].y) * (state->y - turn_circle_intersec[0].y)) - FG_RAD);
	}
	if (nominal[2] >= 0 && nominal[3] >= 0) {  // curve gh
		return fabs(sqrtf((state->x - turn_circle_intersec[1].x) * (state->x - turn_circle_intersec[1].x)
			+ (state->y - turn_circle_intersec[1].y) * (state->y - turn_circle_intersec[1].y)) - GH_RAD);
	}
	if (nominal[4] <= 0 && nominal[5] >= 0) {  // curve hi
		return fabs(sqrtf((state->x - turn_circle_intersec[2].x) * (state->x - turn_circle_intersec[2].x)
			+ (state->y - turn_circle_intersec[2].y) * (state->y - turn_circle_intersec[2].y)) - HI_RAD);
	}
	if (nominal[6] <= 0 && nominal[7] <= 0) {  // curve if
		return fabs(sqrtf((state->x - turn_circle_intersec[3].x) * (state->x - turn_circle_intersec[3].x)
			+ (state->y - turn_circle_intersec[3].y) * (state->y - turn_circle_intersec[3].y)) - IF_RAD);
	}

	// Distance from the tracks (outside of the inner functions)
	if(inner[0] < 0 && nominal[0] <= 0 && nominal[7] >= 0){
	//if (inner[0] < 0 && inner[1] < 0 && inner[2] < 0 && inner[3] > 0) {
		//return fabs(fabs(inner[0] * cosf(atanf(slope[0]))) - 0.5 * ROAD_WIDTH);  // respect to line: f
		return fabs(pointToLineDist(inner[0], inner_fcn[0]) - 0.5 * ROAD_WIDTH);
	}
	if (inner[1] > 0 && nominal[1] >= 0 && nominal[2] <= 0) {
	//if (inner[0] > 0 && inner[1] > 0 && inner[2] < 0 && inner[3] > 0) {
		//return fabs(fabs(inner[1] * cosf(atanf(slope[1]))) - 0.5 * ROAD_WIDTH);  // respect to line: g
		return fabs(pointToLineDist(inner[1], inner_fcn[1]) - 0.5 * ROAD_WIDTH);
	}
	if (inner[2] > 0 && nominal[3] <= 0 && nominal[4] >= 0) {
	//if (inner[0] > 0 && inner[1] < 0 && inner[2] > 0 && inner[3] > 0) {
		//return fabs(fabs(inner[2] * cosf(atanf(slope[2]))) - 0.5 * ROAD_WIDTH);  // respect to line: h
		return fabs(pointToLineDist(inner[2], inner_fcn[2]) - 0.5 * ROAD_WIDTH);
	}
	if (inner[3] < 0 && nominal[5] <= 0 && nominal[6] >= 0) {
	//if (inner[0] > 0 && inner[1] < 0 && inner[2] < 0 && inner[3] < 0) {
		//return fabs(fabs(inner[3] * cosf(atanf(slope[3]))) - 0.5 * ROAD_WIDTH);  // respect to line: i
		return fabs(pointToLineDist(inner[3], inner_fcn[3]) - 0.5 * ROAD_WIDTH);
	}

	 //Distance from the tracks (inside of the inner functions)
	if (inner[0] > 0 && inner[1] < 0 && inner[2] < 0 && inner[3] > 0) {
		float dist_f = fabs(pointToLineDist(inner[0], inner_fcn[0]) + 0.5 * ROAD_WIDTH);
		float dist_g = fabs(pointToLineDist(inner[1], inner_fcn[1]) + 0.5 * ROAD_WIDTH);
		float dist_h = fabs(pointToLineDist(inner[2], inner_fcn[2]) + 0.5 * ROAD_WIDTH);
		float dist_i = fabs(pointToLineDist(inner[3], inner_fcn[3]) + 0.5 * ROAD_WIDTH);
		float min1, min2;
		(dist_f > dist_g) ? min1 = dist_g : min1 = dist_f;
		(dist_h > dist_i) ? min2 = dist_i : min2 = dist_h;
		if (min1 > min2){ 
			return min2; 
		}
		else{ 
			return min1; 
		}
	}

	// If point lies on inner function(s)
	if (inner[0] == 0 || inner[1] == 0 || inner[2] == 0 || inner[3] == 0) {
		float inner_min = 0.0f;
		(fabs(inner[0]) > fabs(inner[1])) ? inner_min = fabs(inner[1]) : inner_min = fabs(inner[0]);
		(inner_min > fabs(inner[2])) ? inner_min = fabs(inner[2]) : inner_min = inner_min;
		(inner_min > fabs(inner[3])) ? inner_min = fabs(inner[3]) : inner_min = inner_min;
		return fabs(inner_min + 0.5 * ROAD_WIDTH);
	}
	return 0;
}

__device__ float distanceFromTrackStraight(float* inner, Track* inner_fcn, Pos* in_intersec, State* state) {
	/*  Calculates the distance from the middle of a designated path
		(Here, from the Tetragon imposing the road infront of the schools library)*/

	float slope[] = { -inner_fcn[0].a, -inner_fcn[1].a, -inner_fcn[2].a, -inner_fcn[3].a };

	// Distance from the curves 
	// @TODO
	if (inner[0] <= 0 && inner[1] >= 0) {  // curve fg
		return fabs(sqrtf((state->x - in_intersec[0].x) * (state->x - in_intersec[0].x)
			+ (state->y - in_intersec[0].y) * (state->y - in_intersec[0].y)) - 0.5 * ROAD_WIDTH);
	}
	if (inner[1] >= 0 && inner[2] >= 0) {  // curve gh
		return fabs(sqrtf((state->x - in_intersec[1].x) * (state->x - in_intersec[1].x)
			+ (state->y - in_intersec[1].y) * (state->y - in_intersec[1].y)) - 0.5 * ROAD_WIDTH);
	}
	if (inner[2] >= 0 && inner[3] <= 0) {  // curve hi
		return fabs(sqrtf((state->x - in_intersec[2].x) * (state->x - in_intersec[2].x)
			+ (state->y - in_intersec[2].y) * (state->y - in_intersec[2].y)) - 0.5 * ROAD_WIDTH);
	}
	if (inner[3] <= 0 && inner[0] <= 0) {  // curve if
		return fabs(sqrtf((state->x - in_intersec[3].x) * (state->x - in_intersec[3].x)
			+ (state->y - in_intersec[3].y) * (state->y - in_intersec[3].y)) - 0.5 * ROAD_WIDTH);
	}

	// Distance from the tracks (outside of the inner functions)
	if (inner[0] <= 0 && inner[1] <= 0 && inner[3] >= 0) {
		//if (inner[0] < 0 && inner[1] < 0 && inner[2] < 0 && inner[3] > 0) {
			//return fabs(fabs(inner[0] * cosf(atanf(slope[0]))) - 0.5 * ROAD_WIDTH);  // respect to line: f
		return fabs(pointToLineDist(inner[0], inner_fcn[0]) - 0.5 * ROAD_WIDTH);
	}
	if (inner[0] >= 0 && inner[1] >= 0 && inner[2] <= 0) {
		//if (inner[0] > 0 && inner[1] > 0 && inner[2] < 0 && inner[3] > 0) {
			//return fabs(fabs(inner[1] * cosf(atanf(slope[1]))) - 0.5 * ROAD_WIDTH);  // respect to line: g
		return fabs(pointToLineDist(inner[1], inner_fcn[1]) - 0.5 * ROAD_WIDTH);
	}
	if (inner[1] <= 0 && inner[2] >= 0 && inner[3] >= 0) {
		//if (inner[0] > 0 && inner[1] < 0 && inner[2] > 0 && inner[3] > 0) {
			//return fabs(fabs(inner[2] * cosf(atanf(slope[2]))) - 0.5 * ROAD_WIDTH);  // respect to line: h
		return fabs(pointToLineDist(inner[2], inner_fcn[2]) - 0.5 * ROAD_WIDTH);
	}
	if (inner[0] >= 0 && inner[2] <= 0 && inner[3] <= 0) {
		//if (inner[0] > 0 && inner[1] < 0 && inner[2] < 0 && inner[3] < 0) {
			//return fabs(fabs(inner[3] * cosf(atanf(slope[3]))) - 0.5 * ROAD_WIDTH);  // respect to line: i
		return fabs(pointToLineDist(inner[3], inner_fcn[3]) - 0.5 * ROAD_WIDTH);
	}

	//Distance from the tracks (inside of the inner functions)
	if (inner[0] > 0 && inner[1] < 0 && inner[2] < 0 && inner[3] > 0) {
		float dist_f = fabs(pointToLineDist(inner[0], inner_fcn[0]) + 0.5 * ROAD_WIDTH);
		float dist_g = fabs(pointToLineDist(inner[1], inner_fcn[1]) + 0.5 * ROAD_WIDTH);
		float dist_h = fabs(pointToLineDist(inner[2], inner_fcn[2]) + 0.5 * ROAD_WIDTH);
		float dist_i = fabs(pointToLineDist(inner[3], inner_fcn[3]) + 0.5 * ROAD_WIDTH);
		float min1, min2;
		(dist_f > dist_g) ? min1 = dist_g : min1 = dist_f;
		(dist_h > dist_i) ? min2 = dist_i : min2 = dist_h;
		if (min1 > min2) {
			return min2;
		}
		else {
			return min1;
		}
	}

	// If point lies on inner function(s)
	if (inner[0] == 0 || inner[1] == 0 || inner[2] == 0 || inner[3] == 0) {
		float inner_min = 0.0f;
		(fabs(inner[0]) > fabs(inner[1])) ? inner_min = fabs(inner[1]) : inner_min = fabs(inner[0]);
		(inner_min > fabs(inner[2])) ? inner_min = fabs(inner[2]) : inner_min = inner_min;
		(inner_min > fabs(inner[3])) ? inner_min = fabs(inner[3]) : inner_min = inner_min;
		return fabs(inner_min + 0.5 * ROAD_WIDTH);
	}
	return 0;
}

__device__ float calculateCost(State* state, Track* outer_fcn, Track* inner_fcn, Track* nominal_fcn, Pos* turn_circle_intersec) {
	/* Calculate the cost of a current state (obstacle collision part added)
	 * index of the path names
	 *          g
	 *        ______
	 *       /     /
	 *    h /     / f
	 *     /     /
	 *     ------
	 *       i
	 * f, g intersection : [x1, y1]
	 * g, h intersection : [x2, y2]
	 * h, i intersection : [x3, y3]
	 * i, f intersection : [x4, y4]*/

	float state_cost = 0.0f;

	float outer[4] = {
		outer_fcn[0].b * state->y + outer_fcn[0].a * state->x + outer_fcn[0].c,
		outer_fcn[1].b * state->y + outer_fcn[1].a * state->x + outer_fcn[1].c,
		outer_fcn[2].b * state->y + outer_fcn[2].a * state->x + outer_fcn[2].c,
		outer_fcn[3].b * state->y + outer_fcn[3].a * state->x + outer_fcn[3].c
	};

	float inner[4] = {
		inner_fcn[0].b* state->y + inner_fcn[0].a * state->x + inner_fcn[0].c,
		inner_fcn[1].b* state->y + inner_fcn[1].a * state->x + inner_fcn[1].c,
		inner_fcn[2].b* state->y + inner_fcn[2].a * state->x + inner_fcn[2].c,
		inner_fcn[3].b* state->y + inner_fcn[3].a * state->x + inner_fcn[3].c
	};

	float nominal[8] = {
		nominal_fcn[0].b* state->y + nominal_fcn[0].a * state->x + nominal_fcn[0].c,
		nominal_fcn[1].b* state->y + nominal_fcn[1].a * state->x + nominal_fcn[1].c,
		nominal_fcn[2].b* state->y + nominal_fcn[2].a * state->x + nominal_fcn[2].c,
		nominal_fcn[3].b* state->y + nominal_fcn[3].a * state->x + nominal_fcn[3].c,
		nominal_fcn[4].b* state->y + nominal_fcn[4].a * state->x + nominal_fcn[4].c,
		nominal_fcn[5].b* state->y + nominal_fcn[5].a * state->x + nominal_fcn[5].c,
		nominal_fcn[6].b* state->y + nominal_fcn[6].a * state->x + nominal_fcn[6].c,
		nominal_fcn[7].b* state->y + nominal_fcn[7].a * state->x + nominal_fcn[7].c
	};

	//float distance = distanceFromTrack(inner_f, inner_g, inner_h, inner_i, inner_fcn);
	float distance = distanceFromTrack(inner, nominal, inner_fcn, turn_circle_intersec, state); // @TODO
	if ((nominal[0] > 0 && nominal[1] < 0) || (nominal[2] > 0 && nominal[3] > 0) || (nominal[4] < 0 && nominal[5] > 0) || (nominal[6] < 0 && nominal[7] < 0)) {
		if (distance > ROAD_WIDTH / 2) {
			state_cost += OFF_ROAD_COST;
		}
	}
	else {
		if ((outer[0] > 0 && outer[1] < 0 && outer[2] < 0 && outer[3]>0) &&
			!(inner[0] > 0 && inner[1] < 0 && inner[2] < 0 && inner[3]>0)) {
			state_cost += 0;
		}
		else {
			state_cost += OFF_ROAD_COST;
		}
	}
	
	state_cost += distance / (ROAD_WIDTH / 2) * TRACK_ERR_COST;
	return state_cost;
}

__device__ float obstacleCollision(State* state, Obs* obstacle) {
	/*  Calculate the relative distace from the middle of the obstacle
		(if output == 0, no collision)*/

	float output = 0.0, obs_fcn = 0.0;
	obs_fcn = powf(state->x - obstacle->x, 2) + powf(state->y - obstacle->y, 2) - powf(obstacle->r, 2);
	if (obs_fcn < 0) {
		output = -obs_fcn / obstacle->r;
	}
	return output;
}

__global__ void initCurand(curandState* state, uint64_t seed) {
	/*	Each thread gets same seed, a different sequence
		number, no offset */

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__global__ void normalRand(curandState* state, float* rand, float scalar) {
	/*  Generate the random number with mean 0.0 and standard
		deviation 1.0, scalar: stdev scalar */
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curandState localState = state[idx];
	rand[idx] = curand_normal(&localState) * scalar;
}

__device__ float getRandom(uint64_t seed, int tid, int shift) {
	curandState s;
	curand_init(seed + tid + shift, 0, 0, &s);
	return curand_normal(&s);
}

__global__ void genControlOld(curandState* state, Control* rand, float* u, float scalar) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// Generate the random number with mean 0.0 and standard deviation 1.0, scalar: stdev scalar
	rand[idx].vd = curand_normal(&state[idx]) * scalar + u[0];
	rand[idx].wd = curand_normal(&state[idx]) * scalar + u[1];
	clampingFcn(&rand[idx]);
}

__global__ void genControl(curandState* state, Control* rand, Control* u, float scalar) {
	/*  Generate V from nominal control U and the generated pertubations E (V = U + E)
		mean 0.0 and standard deviation 1.0, scalar: stdev scalar*/

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curandState localState = state[idx];
	rand[idx].vd = curand_normal(&localState) * scalar + u[threadIdx.x].vd;
	rand[idx].wd = curand_normal(&localState) * scalar + u[threadIdx.x].wd;
	clampingFcn(&rand[idx]);
}

__global__ void genControlNew(uint64_t seed, Control* rand, Control* u, float scalar) {
	/*  Generate V from nominal control U and the generated pertubations E (V = U + E)
		mean 0.0 and standard deviation 1.0, scalar: stdev scalar*/

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	rand[idx].vd = getRandom(seed, idx, 0) * scalar + u[threadIdx.x].vd;
	rand[idx].wd = getRandom(seed, idx, 10) * scalar + u[threadIdx.x].wd;
	clampingFcn(&rand[idx]);
}



__global__ void genState(State* state_list, State* init_state, Control* pert_control) {
	/*  Generate all the states in the prediction horizon for
		each rollout k
		- init_state is k*1 elements long (k-rollouts)
		- need to build k*n list for each prediction state
	*/
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	State state_temp = *init_state;
	for (int n = 0; n < N_HRZ; n++) {
		state_temp = bicycleModel(state_temp, pert_control[n + int(N_HRZ) * idx]); //////???????????????????
		int listidx = n + (N_HRZ)*idx;
		state_list[listidx] = state_temp;
	}

}



__global__ void costFcn_old(float* cost_list, State* state_list,
	Track* outer_fcn, Track* inner_fcn, Obs* obstacle) {
	/*  Calculate all the state costs in the prediction horizon
		for each rollout k */

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float state_cost = 0.0f;
	State state = state_list[idx];

	// Tracking penatlty
	float outer_f = outer_fcn[0].b * state.y + outer_fcn[0].a * state.x + outer_fcn[0].c;
	float outer_g = outer_fcn[1].b * state.y + outer_fcn[1].a * state.x + outer_fcn[1].c;
	float outer_h = outer_fcn[2].b * state.y + outer_fcn[2].a * state.x + outer_fcn[2].c;
	float outer_i = outer_fcn[3].b * state.y + outer_fcn[3].a * state.x + outer_fcn[3].c;

	float inner_f = inner_fcn[0].b * state.y + inner_fcn[0].a * state.x + inner_fcn[0].c;
	float inner_g = inner_fcn[1].b * state.y + inner_fcn[1].a * state.x + inner_fcn[1].c;
	float inner_h = inner_fcn[2].b * state.y + inner_fcn[2].a * state.x + inner_fcn[2].c;
	float inner_i = inner_fcn[3].b * state.y + inner_fcn[3].a * state.x + inner_fcn[3].c;
	float distance = distanceFromTrack_old(inner_f, inner_g, inner_h, inner_i, inner_fcn);

	if ((outer_f > 0 && outer_g < 0 && outer_h < 0 && outer_i>0) &&
		!(inner_f > 0 && inner_g < 0 && inner_h < 0 && inner_i > 0)) {
		state_cost += 0;
	}
	else {
		state_cost += OFF_ROAD_COST;
	}
	state_cost += distance / (ROAD_WIDTH / 2) * TRACK_ERR_COST;

	// Obstacle avoidance penalty

	float collision = 0.0;
	for (int i = 0; obstacle[i].x != NULL; i++) {
		collision = fabs(obstacleCollision(&state, &obstacle[i]));
		state_cost += collision * COLLISION_COST;
	}
	cost_list[idx] = state_cost;
}

__global__ void costFcn(float* cost_list, State* state_list,
	Track* outer_fcn, Track* inner_fcn, Track* nominal_fcn, Obs* obstacle, Pos* turn_circle_intersec) {
	/*  Calculate all the state costs in the prediction horizon
		for each rollout k */
	/* Calculate the cost of a current state (obstacle collision part added)
	* index of the path names
	*          g
	*        ______
	*       /     /
	*    h /     / f
	*     /     /
	*     ------
	*       i
	* f, g intersection : [x1, y1]
	* g, h intersection : [x2, y2]
	* h, i intersection : [x3, y3]
	* i, f intersection : [x4, y4]*/

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float state_cost = 0.0f;
	State state = state_list[idx];
	
	float outer[4] = {
		outer_fcn[0].b * state.y + outer_fcn[0].a * state.x + outer_fcn[0].c,
		outer_fcn[1].b * state.y + outer_fcn[1].a * state.x + outer_fcn[1].c,
		outer_fcn[2].b * state.y + outer_fcn[2].a * state.x + outer_fcn[2].c,
		outer_fcn[3].b * state.y + outer_fcn[3].a * state.x + outer_fcn[3].c
	};

	float inner[4] = {
		inner_fcn[0].b * state.y + inner_fcn[0].a * state.x + inner_fcn[0].c,
		inner_fcn[1].b * state.y + inner_fcn[1].a * state.x + inner_fcn[1].c,
		inner_fcn[2].b * state.y + inner_fcn[2].a * state.x + inner_fcn[2].c,
		inner_fcn[3].b * state.y + inner_fcn[3].a * state.x + inner_fcn[3].c
	};

	float nominal[8] = {
		nominal_fcn[0].b * state.y + nominal_fcn[0].a * state.x + nominal_fcn[0].c,
		nominal_fcn[1].b * state.y + nominal_fcn[1].a * state.x + nominal_fcn[1].c,
		nominal_fcn[2].b * state.y + nominal_fcn[2].a * state.x + nominal_fcn[2].c,
		nominal_fcn[3].b * state.y + nominal_fcn[3].a * state.x + nominal_fcn[3].c,
		nominal_fcn[4].b * state.y + nominal_fcn[4].a * state.x + nominal_fcn[4].c,
		nominal_fcn[5].b * state.y + nominal_fcn[5].a * state.x + nominal_fcn[5].c,
		nominal_fcn[6].b * state.y + nominal_fcn[6].a * state.x + nominal_fcn[6].c,
		nominal_fcn[7].b * state.y + nominal_fcn[7].a * state.x + nominal_fcn[7].c
	};
	float distance = distanceFromTrack(inner, nominal, inner_fcn, turn_circle_intersec, &state);

	if ((nominal[0] >= 0 && nominal[1] <= 0) || (nominal[2] >= 0 && nominal[3] >= 0) || (nominal[4] <= 0 && nominal[5] >= 0) || (nominal[6] <= 0 && nominal[7] <= 0)) {
		if (distance > ROAD_WIDTH / 2) {
			state_cost += OFF_ROAD_COST;
		}
	}
	else {
		if ((outer[0] > 0 && outer[1] < 0 && outer[2] < 0 && outer[3] > 0) &&
			!(inner[0] > 0 && inner[1] < 0 && inner[2] < 0 && inner[3] > 0)) {
			state_cost += 0;
		}
		else {
			state_cost += OFF_ROAD_COST;
		}
	}
	state_cost += distance / (ROAD_WIDTH / 2) * TRACK_ERR_COST;
	
	// Obstacle avoidance penalty

	float collision = 0.0;
	for (int i = 0; obstacle[i].x != NULL; i++) {
		collision = fabs(obstacleCollision(&state, &obstacle[i]));
		state_cost += collision * COLLISION_COST;
    }
	cost_list[idx] = state_cost;
}

__global__ void costFcnStraight(float* cost_list, State* state_list,
	Track* outer_fcn, Track* inner_fcn, Obs* obstacle, Pos* in_intersec) {
	/*  Calculate all the state costs in the prediction horizon
		for each rollout k */
		/* Calculate the cost of a current state (obstacle collision part added)
		* index of the path names
		*          g
		*        ______
		*       /     /
		*    h /     / f
		*     /     /
		*     ------
		*       i
		* f, g intersection : [x1, y1]
		* g, h intersection : [x2, y2]
		* h, i intersection : [x3, y3]
		* i, f intersection : [x4, y4]*/

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float state_cost = 0.0f;
	State state = state_list[idx];

	float outer[4] = {
		outer_fcn[0].b * state.y + outer_fcn[0].a * state.x + outer_fcn[0].c,
		outer_fcn[1].b * state.y + outer_fcn[1].a * state.x + outer_fcn[1].c,
		outer_fcn[2].b * state.y + outer_fcn[2].a * state.x + outer_fcn[2].c,
		outer_fcn[3].b * state.y + outer_fcn[3].a * state.x + outer_fcn[3].c
	};

	float inner[4] = {
		inner_fcn[0].b * state.y + inner_fcn[0].a * state.x + inner_fcn[0].c,
		inner_fcn[1].b * state.y + inner_fcn[1].a * state.x + inner_fcn[1].c,
		inner_fcn[2].b * state.y + inner_fcn[2].a * state.x + inner_fcn[2].c,
		inner_fcn[3].b * state.y + inner_fcn[3].a * state.x + inner_fcn[3].c
	};

	float distance = distanceFromTrackStraight(inner, inner_fcn, in_intersec, &state);

	
	if ((outer[0] > 0 && outer[1] < 0 && outer[2] < 0 && outer[3] > 0) &&
		!(inner[0] > 0 && inner[1] < 0 && inner[2] < 0 && inner[3] > 0)) {
		state_cost += 0;
	}
	else {
		state_cost += OFF_ROAD_COST;
	}
	
	state_cost += distance / (ROAD_WIDTH / 2) * TRACK_ERR_COST;

	// Obstacle avoidance penalty

	float collision = 0.0;
	for (int i = 0; obstacle[i].x != NULL; i++) {
		collision = fabs(obstacleCollision(&state, &obstacle[i]));
		state_cost += collision * COLLISION_COST;
	}
	cost_list[idx] = state_cost;
}

__global__ void calcRolloutCost(float* input, float* output) {
	/*  Calculate the total cost of each rollout and store it into an
		array with reduced size (reduced sum no bank conflict)*/
		// Allocate shared memory
	__shared__ float partial_sum[SHMEM_SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = input[tid];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
	}
}

__global__ void calcRolloutCost2(float* input, float* output) {
	/*  Sequential addressing (reduction sum)*/
	__shared__ float partial_sum[SHMEM_SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = input[tid];
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * threadIdx.x;

		if (index < blockDim.x) {
			partial_sum[index] += partial_sum[index + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
	}
}

__global__ void min_reduction(float* input, float* output) {
	/*  Find the rollout with the minimum cost using sum reduction methods,
		but changing the "sum" part to "min"*/

	__shared__ float partial_sum[K / 32];
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	partial_sum[threadIdx.x] = min(input[i], input[i + blockDim.x]);
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] = min(partial_sum[threadIdx.x], partial_sum[threadIdx.x + s]);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
		//printf("result: %.3f\n", partial_sum[0]);
	}
}

__global__ void min_reduction2(float* input, float* output) {
	/*  Find the rollout with the minimum cost using sum reduction methods,
		but changing the "sum" part to "min"*/

	__shared__ float partial_sum[K / 32];
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = input[i];
	__syncthreads();

	for (int s = blockDim.x / 4; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] = min(partial_sum[threadIdx.x], partial_sum[threadIdx.x + s]);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
	}
}

__global__ void calcWTilde(float* w_tilde, float* rho, float* cost_list) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	w_tilde[idx] = expf(-1 / LAMBDA * (cost_list[idx] - rho[0]));
}

__global__ void genW(Control* u_opt, float* eta, float* w_tilde, Control* V) {
	/*  multiplies the calculated weight of each rollout to its designated predicted controls (N_HRZ)
		u_opt [K,N_HRZ], eta[K], w_tilde[K], V[L, N_HRZ]*/

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float w;
	w = w_tilde[blockIdx.x] / eta[0];
	u_opt[idx].vd = V[idx].vd * w;
	u_opt[idx].wd = V[idx].wd * w;
	//if(blockIdx.x>1000)printf("%d %dw: %.3f\n", blockIdx.x, idx, w);
}

__global__ void genWCorrect(Control* u_opt, float* eta, float* w_tilde, Control* V, Control* U) {
	/*  multiplies the calculated weight of each rollout to the perturbation
		Note that u_opt is completed with U added */

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float w;
	w = w_tilde[blockIdx.x] / eta[0];
	u_opt[idx].vd = (V[idx].vd - U[threadIdx.x].vd) * w;
	u_opt[idx].wd = (V[idx].wd - U[threadIdx.x].wd) * w;
	//if(blockIdx.x>1000)printf("%d %dw: %.3f\n", blockIdx.x, idx, w);
}

__global__ void wsum_reduction(Control* input, Control* output) {
	/*  Calculate the total weighted control of every prediction horizon.
		Input [K,N_HRZ]; Output [N_HRZ]*/

	__shared__ Control partial_sum[1024];

	int tid = threadIdx.x * gridDim.x + blockIdx.x;
	partial_sum[threadIdx.x] = input[tid];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x].vd += partial_sum[threadIdx.x + s].vd;
			partial_sum[threadIdx.x].wd += partial_sum[threadIdx.x + s].wd;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
	}
}

__global__ void sumControl(Control* input, Control* output) {
	/*  Function that sums the input with the output to the output
		(output = input + output)*/

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	output[idx].vd += input[idx].vd;
	output[idx].wd += input[idx].wd;
}


__global__ void wsum_reduction_partial(Control* input, Control* output, int shift) {
	/*  Sum partial (used when the rollout predictions acceed harware constrains, when #threads too large)
		weighted controls of the prediction horizon (sum along columns), can use var shift
		to shift the starting tid*/

	__shared__ Control partial_sum[1024];

	int tid = (threadIdx.x + shift) * gridDim.x + blockIdx.x;
	partial_sum[threadIdx.x] = input[tid];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x].vd += partial_sum[threadIdx.x + s].vd;
			partial_sum[threadIdx.x].wd += partial_sum[threadIdx.x + s].wd;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
	}
}

__global__ void printControl(Control* u) {
	printf("%.3f %.3f\n", u[0].vd, u[0].wd);
}
__global__ void printMinCost(float* c) {
	printf("min cost: %.3f\n", c[0]);
}

__global__ void line() {
	printf("=====================\n");
}

__global__ void printfloat(float* f) {
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	printf("float: %.3f\n", f[i]);
}

__global__ void PertCost(float* costList, Control* rand, Control* u) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	costList[i] += RHO * (u[threadIdx.x].vd * RAND_SCALAR * rand[i].vd + u[threadIdx.x].wd * RAND_SCALAR * rand[i].wd);
}

__host__ Pos getIntersec(Track line1, Track line2) {
	/*  Calculate the intersection point of two line fcns*/
	Pos intersec;
	intersec.x = (line1.b * line2.c - line2.b * line1.c) / (line1.a * line2.b - line2.a * line1.b);
	intersec.y = (line1.c * line2.a - line2.c * line1.a) / (line1.a * line2.b - line2.a * line1.b);
	return intersec;
}

__host__ void build_track_old(Track* inner_fcn, Track* outer_fcn, Pos* mid_intersec, Pos* inner_intersec,
	Pos* outer_intersec, Track* mid_fcn) {
	/*  Build the boundaries of a designated path, also return
		the intersection points of the middle path and boundaries
	 *  Index of the path names
	 *          g
	 *        ______
	 *       /     /
	 *    h /     / f
	 *     /     /
	 *     ------
	 *        i
	 *  f, g intersection : [x1, y1] :0
	 *  g, h intersection : [x2, y2] :1
	 *  h, i intersection : [x3, y3] :2
	 *  i, f intersection : [x4, y4] :3 */

	float mid_m[4] = { -mid_fcn[0].a, -mid_fcn[1].a, -mid_fcn[2].a, -mid_fcn[3].a };

	memcpy(outer_fcn, mid_fcn, 4 * sizeof(Track));
	outer_fcn[0].c += (ROAD_WIDTH / 2) / cos(atan(mid_m[0]));
	outer_fcn[1].c -= (ROAD_WIDTH / 2) / cos(atan(mid_m[1]));
	outer_fcn[2].c -= (ROAD_WIDTH / 2) / cos(atan(mid_m[2]));
	outer_fcn[3].c += (ROAD_WIDTH / 2) / cos(atan(mid_m[3]));

	memcpy(inner_fcn, mid_fcn, 4 * sizeof(Track));
	inner_fcn[0].c -= (ROAD_WIDTH / 2) / cos(atan(mid_m[0]));
	inner_fcn[1].c += (ROAD_WIDTH / 2) / cos(atan(mid_m[1]));
	inner_fcn[2].c += (ROAD_WIDTH / 2) / cos(atan(mid_m[2]));
	inner_fcn[3].c -= (ROAD_WIDTH / 2) / cos(atan(mid_m[3]));

	mid_intersec[0] = getIntersec(mid_fcn[0], mid_fcn[1]);
	mid_intersec[1] = getIntersec(mid_fcn[1], mid_fcn[2]);
	mid_intersec[2] = getIntersec(mid_fcn[2], mid_fcn[3]);
	mid_intersec[3] = getIntersec(mid_fcn[3], mid_fcn[0]);

	inner_intersec[0] = getIntersec(inner_fcn[0], inner_fcn[1]);
	inner_intersec[1] = getIntersec(inner_fcn[1], inner_fcn[2]);
	inner_intersec[2] = getIntersec(inner_fcn[2], inner_fcn[3]);
	inner_intersec[3] = getIntersec(inner_fcn[3], inner_fcn[0]);

	outer_intersec[0] = getIntersec(outer_fcn[0], outer_fcn[1]);
	outer_intersec[1] = getIntersec(outer_fcn[1], outer_fcn[2]);
	outer_intersec[2] = getIntersec(outer_fcn[2], outer_fcn[3]);
	outer_intersec[3] = getIntersec(outer_fcn[3], outer_fcn[0]);
}

__host__ Track getNominal(Track line, Pos point) {
	Track nominal;
	nominal.a = -1 / line.a;
	nominal.b = line.b;
	nominal.c = -(nominal.a * point.x + nominal.b * point.y);
	return nominal;
}

__host__ void build_track(Track* inner_fcn, Track* outer_fcn, Pos* mid_intersec, Pos* inner_intersec,
	Pos* outer_intersec, Track* mid_fcn, Pos* turn_center_intersec, Track* nominal_fcn) {
	/*  Build the boundaries of a designated path, also return
	the intersection points of the middle path and boundaries
 *  Index of the path names
 *          g
 *        ______
 *       /     /
 *    h /     / f
 *     /     /
 *     ------
 *        i
 *  f, g intersection : [x1, y1] f: 0
 *  g, h intersection : [x2, y2] g: 1
 *  h, i intersection : [x3, y3] h: 2
 *  i, f intersection : [x4, y4] i: 3 */

	float mid_m[4] = { -mid_fcn[0].a, -mid_fcn[1].a, -mid_fcn[2].a, -mid_fcn[3].a };

	memcpy(outer_fcn, mid_fcn, 4 * sizeof(Track));
	outer_fcn[0].c += (ROAD_WIDTH / 2) / cos(atan(mid_m[0]));
	outer_fcn[1].c -= (ROAD_WIDTH / 2) / cos(atan(mid_m[1]));
	outer_fcn[2].c -= (ROAD_WIDTH / 2) / cos(atan(mid_m[2]));
	outer_fcn[3].c += (ROAD_WIDTH / 2) / cos(atan(mid_m[3]));

	memcpy(inner_fcn, mid_fcn, 4 * sizeof(Track));
	inner_fcn[0].c -= (ROAD_WIDTH / 2) / cos(atan(mid_m[0]));
	inner_fcn[1].c += (ROAD_WIDTH / 2) / cos(atan(mid_m[1]));
	inner_fcn[2].c += (ROAD_WIDTH / 2) / cos(atan(mid_m[2]));
	inner_fcn[3].c -= (ROAD_WIDTH / 2) / cos(atan(mid_m[3]));

	mid_intersec[0] = getIntersec(mid_fcn[0], mid_fcn[1]);
	mid_intersec[1] = getIntersec(mid_fcn[1], mid_fcn[2]);
	mid_intersec[2] = getIntersec(mid_fcn[2], mid_fcn[3]);
	mid_intersec[3] = getIntersec(mid_fcn[3], mid_fcn[0]);

	inner_intersec[0] = getIntersec(inner_fcn[0], inner_fcn[1]);
	inner_intersec[1] = getIntersec(inner_fcn[1], inner_fcn[2]);
	inner_intersec[2] = getIntersec(inner_fcn[2], inner_fcn[3]);
	inner_intersec[3] = getIntersec(inner_fcn[3], inner_fcn[0]);

	outer_intersec[0] = getIntersec(outer_fcn[0], outer_fcn[1]);
	outer_intersec[1] = getIntersec(outer_fcn[1], outer_fcn[2]);
	outer_intersec[2] = getIntersec(outer_fcn[2], outer_fcn[3]);
	outer_intersec[3] = getIntersec(outer_fcn[3], outer_fcn[0]);

	// Setup the curves
	Track fg_inner_tmp[2] = { mid_fcn[0], mid_fcn[1] },
		gh_inner_tmp[2] = { mid_fcn[1], mid_fcn[2] },
		hi_inner_tmp[2] = { mid_fcn[2], mid_fcn[3] },
		if_inner_tmp[2] = { mid_fcn[3], mid_fcn[0] };

	// Shift the mid fcn inwards by the turning radius
	fg_inner_tmp[0].c -= (FG_RAD) / cos(atan(mid_m[0]));
	fg_inner_tmp[1].c += (FG_RAD) / cos(atan(mid_m[1]));

	gh_inner_tmp[0].c += (GH_RAD) / cos(atan(mid_m[1]));
	gh_inner_tmp[1].c += (GH_RAD) / cos(atan(mid_m[2]));

	hi_inner_tmp[0].c += (HI_RAD) / cos(atan(mid_m[2]));
	hi_inner_tmp[1].c -= (HI_RAD) / cos(atan(mid_m[3]));

	if_inner_tmp[0].c -= (IF_RAD) / cos(atan(mid_m[3]));
	if_inner_tmp[1].c -= (IF_RAD) / cos(atan(mid_m[0]));

	// Get the center of the turning curves
	turn_center_intersec[0] = getIntersec(fg_inner_tmp[0], fg_inner_tmp[1]);
	turn_center_intersec[1] = getIntersec(gh_inner_tmp[0], gh_inner_tmp[1]);
	turn_center_intersec[2] = getIntersec(hi_inner_tmp[0], hi_inner_tmp[1]);
	turn_center_intersec[3] = getIntersec(if_inner_tmp[0], if_inner_tmp[1]);

	nominal_fcn[0] = getNominal(mid_fcn[0], turn_center_intersec[0]);
	nominal_fcn[1] = getNominal(mid_fcn[1], turn_center_intersec[0]);
	nominal_fcn[2] = getNominal(mid_fcn[1], turn_center_intersec[1]);
	nominal_fcn[3] = getNominal(mid_fcn[2], turn_center_intersec[1]);
	nominal_fcn[4] = getNominal(mid_fcn[2], turn_center_intersec[2]);
	nominal_fcn[5] = getNominal(mid_fcn[3], turn_center_intersec[2]);
	nominal_fcn[6] = getNominal(mid_fcn[3], turn_center_intersec[3]);
	nominal_fcn[7] = getNominal(mid_fcn[0], turn_center_intersec[3]);
}

main() {


	// Record runtime 
	cudaEvent_t start, stop;
	float rec_time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Setup parameters and Initialize variables
	Control* host_V, * dev_V;
	Control* host_U, * dev_U;
	Control output_u0;
	Control* dev_u_opt, * dev_u_opt_part, * dev_u_opt_part2, * dev_u_opt_part3, * host_u_opt;
	Control* dev_temp_u_opt;
	//State host_x0 = { 150.91,126.71,-1,2 };
	//State host_x0 = { 136, 138,-1,2 };
	State host_x0 = { 128, 144,-1,2 };
	//State host_x0 = { 7,31.5, 2.5, 2.726 };
	State* dev_x0;
	State* dev_stateList, * host_stateList;
	//curandState* dev_cstate;
	float* dev_state_costList;
	float* dev_rollout_costList;
	float* host_rollout_costList;
	float* dev_rho;
	float* dev_w_tilde;
	float* dev_eta;

	//Build obstacle
	Obs host_obstacle[] = {
		{154 ,126.2 ,1},
		{153 ,124.5 ,1},
		{152 ,120.5 ,1},
		{152 ,119   ,1}
	};
	//Obs host_obstacle[] = { 0 };
	Obs* dev_obstacle;
	int NUM_OBS = sizeof(host_obstacle) / sizeof(Obs);

	// Build track
	Track host_mid_fcn[] = {
		{1.0000, -1.33200, 82.62978},
		{1.0000, 0.75640, -240.86623},
		{1.0000, -1.36070, -33.13473},
		{1.0000, 0.47203, -35.00739}
	};
	Track* host_in_fcn, * host_out_fcn, * host_nominal_fcn;
	Track* dev_mid_fcn, * dev_in_fcn, * dev_out_fcn, *dev_nominal_fcn;
	Pos* host_mid_intersec, * host_in_intersec, * host_out_intersec, * host_turn_circle_intersec;
	Pos* dev_mid_intersec, * dev_in_intersec, * dev_out_intersec, * dev_turn_circle_intersec;

	// Setup host memory
	host_V = (Control*)malloc(int(K * N_HRZ) * sizeof(Control));
	host_u_opt = (Control*)malloc(int(N_HRZ) * sizeof(Control));
	host_stateList = (State*)malloc(int(K * N_HRZ) * sizeof(State));
	host_in_fcn = (Track*)malloc(4 * sizeof(Track));
	host_out_fcn = (Track*)malloc(4 * sizeof(Track));
	host_mid_intersec = (Pos*)malloc(4 * sizeof(Pos));
	host_in_intersec = (Pos*)malloc(4 * sizeof(Pos));
	host_out_intersec = (Pos*)malloc(4 * sizeof(Pos));
	host_U = (Control*)malloc(int(N_HRZ) * sizeof(Control));
	host_rollout_costList = (float*)malloc(int(K) * sizeof(float));
	host_nominal_fcn = (Track*)malloc(8 * sizeof(Track));
	host_turn_circle_intersec = (Pos*)malloc(4 * sizeof(Pos));

	// Setup device memory
	//cudaMalloc((void**)&/*dev_cstate*/, int(K * N_HRZ) * sizeof(curandState));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_V, int(K * N_HRZ) * sizeof(Control)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_x0, sizeof(State)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_stateList, int(K * N_HRZ) * sizeof(State)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_state_costList, int(K * N_HRZ) * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_rollout_costList, int(K) * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_mid_fcn, 4 * sizeof(Track)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_in_fcn, 4 * sizeof(Track)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_out_fcn, 4 * sizeof(Track)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_nominal_fcn, 8 * sizeof(Track)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_mid_intersec, 4 * sizeof(Pos)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_in_intersec, 4 * sizeof(Pos)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_out_intersec, 4 * sizeof(Pos)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_turn_circle_intersec, 4 * sizeof(Pos)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_rho, int(K) * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_w_tilde, int(K) * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_eta, int(K) * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_u_opt, int(N_HRZ) * sizeof(Control)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_u_opt_part, int(N_HRZ) * sizeof(Control)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_u_opt_part2, int(N_HRZ) * sizeof(Control)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_u_opt_part3, int(N_HRZ) * sizeof(Control)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_temp_u_opt, int(K * N_HRZ) * sizeof(Control)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_U, int(N_HRZ) * sizeof(Control)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_obstacle, NUM_OBS * sizeof(Obs)));

	// Setup constant memory
	//build_track(host_in_fcn, host_out_fcn, host_mid_intersec, host_in_intersec, host_out_intersec, host_mid_fcn);
	build_track(host_in_fcn, host_out_fcn, host_mid_intersec, host_in_intersec, host_out_intersec, host_mid_fcn, host_turn_circle_intersec, host_nominal_fcn);
	CUDA_SAFE_CALL(cudaMemcpy(dev_mid_fcn, host_mid_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_in_fcn, host_in_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_out_fcn, host_out_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_nominal_fcn, host_nominal_fcn, 8 * sizeof(Track), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_mid_intersec, host_mid_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_in_intersec, host_in_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_out_intersec, host_out_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_turn_circle_intersec, host_turn_circle_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_obstacle, host_obstacle, NUM_OBS * sizeof(Obs), cudaMemcpyHostToDevice));

	// Initialize nominal conrtrol
	for (int n = 0; n < N_HRZ; n++) {
		//host_U[n].vd = host_u0[0];
		//host_U[n].wd = host_u0[1];
		host_U[n].vd = 0;
		host_U[n].wd = 0;
	}

	// Output .csv files
	std::fstream outputFile;
	std::fstream testFile;
	std::fstream cost;
	std::fstream predFile;

	// create a name for the file output
	outputFile.open("MPC_output11.csv", ios::out | ios::app);
	testFile.open("testFile.csv", ios::out | ios::app);
	cost.open("costroll.csv", ios::out | ios::app);
	predFile.open("predstate.csv", ios::out | ios::app);

	// Initialize random seed outside loop (will generate same random numbers but greatly increase efficiency)

	//cout << (int)time(NULL);
	//initCurand << <N_HRZ * 8, K / 8 >> > (dev_cstate, uint64_t(time(NULL)));

	// Calculate average runtime 
	double runtime_avg = 0;
	double runtime_list[ITERATIONS] = { 0 };
	
	

	// Start actual MPC iteration
	for (int it = 0; it < ITERATIONS; it++) {
		auto start = chrono::steady_clock::now();
		CUDA_SAFE_CALL(cudaMemcpy(dev_U, host_U, int(N_HRZ) * sizeof(Control), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(dev_x0, &host_x0, sizeof(State), cudaMemcpyHostToDevice));

		// Launch kernal functions
		//cudaEventRecord(start, 0);

		//initCurand << <N_HRZ * 8, K / 8 >> > (dev_cstate, uint64_t(time(NULL)));/// (long)it);  // Slow! might not want it in loop?
		//cudaEventRecord(start, 0);

		//genControl << <K, N_HRZ >> > (dev_cstate, dev_V, dev_U, RAND_SCALAR);
		//cout << 1000*time(NULL) << endl;
		genControlNew << <K, int(N_HRZ) >> > (time(NULL), dev_V, dev_U, RAND_SCALAR);

		genState << <K / 256, K / 4 >> > (dev_stateList, dev_x0, dev_V);

		costFcn << <int(N_HRZ) * 4, K / 4 >> > (dev_state_costList, dev_stateList, dev_out_fcn, dev_in_fcn, dev_nominal_fcn, dev_obstacle, dev_turn_circle_intersec);
		//costFcnStraight << <int(N_HRZ) * 4, K / 4 >> > (dev_state_costList, dev_stateList, dev_out_fcn, dev_in_fcn, dev_obstacle, dev_in_intersec);
		//costFcn_old << <int(N_HRZ) * 4, K / 4 >> > (dev_state_costList, dev_stateList, dev_out_fcn, dev_in_fcn, dev_obstacle);
		PertCost << <K, int(N_HRZ) >> > (dev_state_costList, dev_V, dev_U);

		calcRolloutCost2 << <K, int(N_HRZ) >> > (dev_state_costList, dev_rollout_costList);

		min_reduction << <K / 32 / 2, K / 32 >> > (dev_rollout_costList, dev_rho);
		min_reduction2 << <1, K / 32 >> > (dev_rho, dev_rho);

		//printMinCost << <1, 1 >> > (dev_rho);

		calcWTilde << <K / 32, K / 32 >> > (dev_w_tilde, dev_rho, dev_rollout_costList);

		sum_reduction << <K / 32 / 2, K / 32 >> > (dev_w_tilde, dev_eta);

		sum_reduction << <1, K / 32 >> > (dev_eta, dev_eta);

		//genW << <K, int(N_HRZ) >> > (dev_temp_u_opt, dev_eta, dev_w_tilde, dev_V);
		genWCorrect << <K, int(N_HRZ) >> > (dev_temp_u_opt, dev_eta, dev_w_tilde, dev_V, dev_U);

		wsum_reduction << <int(N_HRZ), K >> > (dev_temp_u_opt, dev_u_opt);

		// If 1024 threads are too much for the hardware, the method below can be adapted
		// <<< N_HRZ, K(max threads(512, 1024, 2048)) >>>
		//wsum_reduction_partial << <40, 512 >> > (dev_temp_u_opt, dev_u_opt, 0);
		//wsum_reduction_partial << <40, 512 >> > (dev_temp_u_opt, dev_u_opt_part,  512);
		//wsum_reduction_partial << <40, 512 >> > (dev_temp_u_opt, dev_u_opt_part2, 1024);
		//wsum_reduction_partial << <40, 512 >> > (dev_temp_u_opt, dev_u_opt_part3, 1536);
		//sumControl << <20, 20 >> > (dev_u_opt_part, dev_u_opt);
		//sumControl << <20, 20 >> > (dev_u_opt_part2, dev_u_opt);
		//sumControl << <20, 20 >> > (dev_u_opt_part3, dev_u_opt);

		//cudaEventRecord(stop, 0);

		//// Show runtime in miliseconds
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&rec_time, start, stop);
		//printf("kernal runtime: %.5f ms\n", rec_time);
		//cudaEventDestroy(start);
		//cudaEventDestroy(stop);

		// Acts as a syncing buffer so no need for additional synhronization
		CUDA_SAFE_CALL(cudaMemcpy(host_V, dev_V, int(K * N_HRZ) * sizeof(Control), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(host_stateList, dev_stateList, int(K * N_HRZ) * sizeof(State), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(host_u_opt, dev_u_opt, int(N_HRZ) * sizeof(Control), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(host_rollout_costList, dev_rollout_costList, int(K) * sizeof(float), cudaMemcpyDeviceToHost));

		// //Store rollout data to csv, which can be visualized using Matlab
		//for (int j = 0; j < 40 * 1024; j++) {
		//	testFile << host_stateList[j].x << ",";
		//}
		//testFile << std::endl;
		//for (int j = 0; j < 40 * 1024; j++) {
		//	testFile << host_stateList[j].y << ",";
		//}
		//testFile << std::endl;
		//for (int j = 0; j < 40 * 1024; j++) {
		//	testFile << host_stateList[j].phi << ",";
		//}
		//testFile << std::endl;
		//for (int j = 0; j < 40 * 1024; j++) {
		//	testFile << host_stateList[j].v0 << ",";
		//}
		//testFile << std::endl;

		//// Store rollout cost (K) to a apecific csv file
		//for (int j = 0; j < 1024; j++) {
		//	cost << host_rollout_costList[j] << std::endl;
		//}

		//// Store the weighted control sequence and its predicted states to a csv file
		//State predstate[40];
		//predstate[0] = host_x0;
		//for (int i = 0; i < N_HRZ; i++) {
		//	int count = i - 1;
		//	if (count < 0) count = 0;
		//	predstate[i] = bicycleModel(predstate[count], host_u_opt[i]);
		//}
		//for (int j = 0; j < 40; j++) {
		//	predFile << predstate[j].x << ",";
		//}
		//predFile << std::endl;
		//for (int j = 0; j < 40; j++) {
		//	predFile << predstate[j].y << ",";
		//}
		//predFile << std::endl;
		//for (int j = 0; j < 40; j++) {
		//	predFile << predstate[j].phi << ",";
		//}
		//predFile << std::endl;
		//for (int j = 0; j < 40; j++) {
		//	predFile << predstate[j].v0 << ",";
		//}
		//predFile << std::endl;
		//for (int j = 0; j < 40; j++) {
		//	predFile << host_u_opt[j].vd << ",";
		//}
		//predFile << std::endl;
		//for (int j = 0; j < 40; j++) {
		//	predFile << host_u_opt[j].wd << ",";
		//}
		//predFile << std::endl;



		//// Not really important here
		//printf("returned random value is %.1f %.1f %.1f\n", host_V[0].vd, host_V[1].vd, host_V[2].wd);
		//printf("returned state list value is %.1f %.1f %.1f %.1f\n", host_stateList[1].x, host_stateList[1].y, host_stateList[1].phi, host_stateList[1].v0);
		//printf("returned state list value is %.1f %.1f %.1f %.1f\n", host_stateList[5].x, host_stateList[5].y, host_stateList[5].phi, host_stateList[5].v0);
		//printf("vd: ");
		//for (int i = 0; i < N_HRZ - 15; i++) {
		//	printf("%2.2f ", host_u_opt[i].vd);
		//}
		//printf("\nwd: ");
		//for (int i = 0; i < N_HRZ - 15; i++) {
		//	printf("%2.2f ", host_u_opt[i].wd);
		//}
		//printf("\n");

		// if use genWCorrect
		for (int i = 0; i < N_HRZ; i++) {
			host_u_opt[i].vd += host_U[i].vd;
			host_u_opt[i].wd += host_U[i].wd;
		}

		// Generate the predicted next state with u_opt
		host_x0 = bicycleModel(host_x0, host_u_opt[0]);

		// Store state data to csv and see results by plotting data in excel
		//printf("%.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", host_x0.x, host_x0.y, host_x0.phi, host_x0.v0, host_u_opt[0].vd, host_u_opt[0].wd);
		outputFile << it << "," << host_x0.x << "," << host_x0.y << "," << host_x0.phi << "," << host_x0.v0 << "," << host_u_opt[0].vd << "," << host_u_opt[0].wd << std::endl;

		// Shift the control by 1 
		output_u0 = host_u_opt[0];
		memmove(host_u_opt, &host_u_opt[1], int(N_HRZ - 1) * sizeof(Control)); // use memmove instead of memcpy because the destination overlaps the source
		host_u_opt[int(N_HRZ)] = host_u_opt[int(N_HRZ) - 1];
		memcpy(host_U, host_u_opt, int(N_HRZ) * sizeof(Control));

		// Record the time of one oteration
		cudaDeviceSynchronize();  //test 
		auto end = chrono::steady_clock::now();
		auto diff = end - start;
		cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;
		runtime_list[it] = chrono::duration <double, milli>(diff).count();
	}

	// Average the runtime over all the iterations
	for (int i = 0; i < ITERATIONS; i++) {
		runtime_avg += runtime_list[i];
	}
	runtime_avg /= double(ITERATIONS);
	cout << "AVG Runtime: " << runtime_avg << " (ms)" << endl;

	// Close cvs files after record
	outputFile.close();
	testFile.close();
	cost.close();

	return 0;
}
