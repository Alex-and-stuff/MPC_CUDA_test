#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <curand_kernel.h>
#include <curand.h>

#define SIZE		256
#define SHMEM_SIZE	256 
#define DSIZE		102400
#define USIZE		40

#define CONTROLSIZE 2				// Number of controls u = [vd,wd]
#define STATESIZE   4				// Number of states   x = [x,y,phi,v0]
#define T_HRZ       2.0f			// Prediction time hoeizon
#define F_STP		20.0f			// Frequency of prediction (control freq)
#define T_STP		1/F_STP			// Period of prediction
#define N_HRZ       T_HRZ*F_STP		// N steps in a time horizon
#define K			1024            // K-rollout predictions
#define V_MAX       4.0f			// Velocity command upper-bound
#define V_MIN       1.5f			// Velocity command lower-bound
#define W_MAX		1.0f			// Angular acceleration command upper-bound
#define W_MIN		-1.0f			// Angular acceleration command lower-bound
#define ROAD_WIDTH  2.0f			// Width of road (unit: m)
#define LAMBDA      50.0f

#define OFF_ROAD_COST	500.0f		// Penalty for leaving the road
#define COLLISION_COST	800.0f		// Penalty for colliding into an obstacle
#define TRACK_ERR_COST	5.0f		// Penalty for tracking error (parting from the middle)

struct Control {
	float vd, wd, padding;
};
struct State {
	float x, y, phi, v0, padding;
};
struct Track {
	// Track fcn in the form of by + ax + c =0
	float b, a, c;
};
struct Pos {
	float x, y;
};

__global__ void sum_reduction(float* v, float* v_r) {
	// Allocate shared memory
	__shared__ float partial_sum[64];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
		//printf("result: %.3f\n", partial_sum[0]);
		//printf("%4d::%.3f\n", blockIdx.x, v_r[blockIdx.x]);
	}
}

__global__ void printVar(float* var) {
	printf("====%.3f====", var[0]);
}

void initialize_vector(int* v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = 2;//rand() % 10;
	}
}

__device__ State bicycleModel(State state, Control u) {
	State state_dot;
	State state_next;
	float Ka = 2.54;
	//printf("%.2f %.2f\n", u.vd, u.wd);
	state_dot.x = state.v0 * cosf(state.phi);
	state_dot.y = state.v0 * sinf(state.phi);
	state_dot.phi = u.wd;
	state_dot.v0 = Ka * state.v0 * (u.vd - state.v0);
	//printf("%.1f %.1f %.1f %.1f\n", state_dot.x, state_dot.y, state_dot.phi, state_dot.v0);
	state_next.x = state.x + state_dot.x * T_STP;
	state_next.y = state.y + state_dot.y * T_STP;
	state_next.phi = state.phi + state_dot.phi * T_STP;
	state_next.v0 = state.v0 + state_dot.v0 * T_STP;
	//printf("%.1f %.1f %.1f %.1f\n", state_next.x, state_next.y, state_next.phi, state_next.v0);
	return state_next;
}

__device__ Control clampingFcn(Control* u_in) {
	/*	Clamping fcn for the perturbated command, acting as a
		hard contrain*/
	if (u_in->vd > V_MAX) { u_in->vd = V_MAX; }
	else if (u_in->vd < V_MIN) { u_in->vd = V_MIN; }
	if (u_in->wd > W_MAX) { u_in->wd = W_MAX; }
	else if (u_in->wd < W_MIN) { u_in->wd = W_MIN; }
	//printf("out %.2f %.2f\n", u_in->vd, u_in->wd);
}

__device__ float distanceFromTrack(float inner_f, float inner_g, float inner_h,
	float inner_i, Track* inner_fcn) {
	float slope[] = { -inner_fcn[0].a, -inner_fcn[1].a, -inner_fcn[2].a, -inner_fcn[3].a };
	//printf("%.3f %.3f %.3f %.3f\n", slope[0], slope[1], slope[2], slope[3]);
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
	return distance;
}

__device__ float calculateCost(State* state, Track* outer_fcn, Track* inner_fcn) {
	float state_cost = 0.0f;
	float outer_f = outer_fcn[0].b * state->y + outer_fcn[0].a * state->x + outer_fcn[0].c;
	float outer_g = outer_fcn[1].b * state->y + outer_fcn[1].a * state->x + outer_fcn[1].c;
	float outer_h = outer_fcn[2].b * state->y + outer_fcn[2].a * state->x + outer_fcn[2].c;
	float outer_i = outer_fcn[3].b * state->y + outer_fcn[3].a * state->x + outer_fcn[3].c;

	float inner_f = inner_fcn[0].b * state->y + inner_fcn[0].a * state->x + inner_fcn[0].c;
	float inner_g = inner_fcn[1].b * state->y + inner_fcn[1].a * state->x + inner_fcn[1].c;
	float inner_h = inner_fcn[2].b * state->y + inner_fcn[2].a * state->x + inner_fcn[2].c;
	float inner_i = inner_fcn[3].b * state->y + inner_fcn[3].a * state->x + inner_fcn[3].c;

	float distance = distanceFromTrack(inner_f, inner_g, inner_h, inner_i, inner_fcn);
	if ((outer_f > 0 && outer_g < 0 && outer_h < 0 && outer_i>0) &&
		!(inner_f > 0 && inner_g < 0 && inner_h < 0 && inner_i>0)) {
		state_cost += 0;
	}
	else {
		state_cost += OFF_ROAD_COST;
	}
	state_cost += distance / (ROAD_WIDTH / 2) * TRACK_ERR_COST;
	return state_cost;
}

__global__ void initCurand(curandState* state, unsigned long seed) {
	/*	Each thread gets same seed, a different sequence
		number, no offset */
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__global__ void normalRand(curandState* state, float* rand, float scalar) {
	/*  Generate the random number with mean 0.0 and standard
		deviation 1.0, scalar: stdev scalar */
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	rand[idx] = curand_normal(&state[idx]) * scalar;
}

__global__ void genControlOld(curandState* state, Control* rand, float* u, float scalar) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// Generate the random number with mean 0.0 and standard deviation 1.0, scalar: stdev scalar
	rand[idx].vd = curand_normal(&state[idx]) * scalar + u[0];
	rand[idx].wd = curand_normal(&state[idx]) * scalar + u[1];
	clampingFcn(&rand[idx]);
	//printf("%.1f\n", rand[idx].v);
}

__global__ void genControl(curandState* state, Control* rand, Control* u, float scalar) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// Generate the random number with mean 0.0 and standard deviation 1.0, scalar: stdev scalar
	//printf("%.3f %.3f\n", u[threadIdx.x].vd, u[threadIdx.x].wd);
	//printf("%d\n", idx);
	rand[idx].vd = curand_normal(&state[idx]) * scalar + u[threadIdx.x].vd;
	rand[idx].wd = curand_normal(&state[idx]) * scalar + u[threadIdx.x].wd;
	clampingFcn(&rand[idx]);
	//printf("%.3f %.3f %.3f\n", curand_normal(&state[idx]) * scalar, u[threadIdx.x].vd, rand[idx].vd);
}

__global__ void genState(State* state_list, State* init_state, Control* pert_control) {
	/*  Generate all the states in the prediction horizon for
		each rollout k
		- init_state is k*1 elements long (k-rollouts)
		- need to build k*n list for each prediction state
	*/
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	State state_temp = *init_state;
	//float cost = 0;
	//printf("%.1f %.1f %.1f %.1f\n", init_state->x, init_state->y, init_state->phi, init_state->v0);
	for (int n = 0; n < N_HRZ; n++) {
		//printf("%.3f %.3f\n", pert_control[idx + n * idx].vd, pert_control[idx + n * idx].wd);
		state_temp = bicycleModel(state_temp, pert_control[n + n * idx]);
		int listidx = n + (N_HRZ) * idx;
		//printf("%d::: %.1f %.1f %.1f %.1f\n", n, state_temp.x, state_temp.y, state_temp.phi, state_temp.v0);
		state_list[listidx] = state_temp;
		//if (listidx <10) printf("%d:::%.2f %.2f %.2f %.2f\n", listidx, state_list[listidx].x, state_list[listidx].y, state_list[listidx].phi, state_list[listidx].v0);
		//printf("%d:::%.2f %.2f %.2f %.2f\n", idx + n * idx, state_list[idx + n * idx].x, state_list[idx + n * idx].y, state_list[idx + n * idx].phi, state_list[idx + n * idx].v0);
		//printf("%.1f ",state_list[idx + n * idx].x);
		//printf("%.2f %.2f\n", pert_control[idx + n * idx].vd, pert_control[idx + n * idx].wd);
		//cost += calculateCost(&state_temp, outer_fcn, inner_fcn);
	}
	
}

__global__ void test(State* state_list) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx<10) printf("%d:  %.3f %.3f %.3f %.3f\n", idx, state_list[idx].x, state_list[idx].y, state_list[idx].phi, state_list[idx].v0);
}

__global__ void costFcn(float* cost_list, State* state_list,  
	Track* outer_fcn, Track* inner_fcn) {
	/*  Calculate all the state costs in the prediction horizon 
		for each rollout k */
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float state_cost = 0.0f;
	State state = state_list[idx];
	//printf("%.3f %.3f %.3f\n", inner_fcn[3].b, inner_fcn[3].a, inner_fcn[3].c);
	//printf("%.3f %.3f %.3f\n", outer_fcn[0].b, outer_fcn[0].a, outer_fcn[0].c);
	//if(idx>40000) printf("%d\n", idx);
	//printf("%d:  %.3f %.3f %.3f %.3f\n", idx, state_list[idx].x, state_list[idx].y, state_list[idx].phi, state_list[idx].v0);  ///////////////////////////////////////////// something wrong here
	float outer_f = outer_fcn[0].b * state.y + outer_fcn[0].a * state.x + outer_fcn[0].c;  
	float outer_g = outer_fcn[1].b * state.y + outer_fcn[1].a * state.x + outer_fcn[1].c;
	float outer_h = outer_fcn[2].b * state.y + outer_fcn[2].a * state.x + outer_fcn[2].c;
	float outer_i = outer_fcn[3].b * state.y + outer_fcn[3].a * state.x + outer_fcn[3].c;

	float inner_f = inner_fcn[0].b * state.y + inner_fcn[0].a * state.x + inner_fcn[0].c;
	float inner_g = inner_fcn[1].b * state.y + inner_fcn[1].a * state.x + inner_fcn[1].c;
	float inner_h = inner_fcn[2].b * state.y + inner_fcn[2].a * state.x + inner_fcn[2].c;
	float inner_i = inner_fcn[3].b * state.y + inner_fcn[3].a * state.x + inner_fcn[3].c;

	//printf("%.3f %.3f %.3f %.3f\n", inner_f, inner_g, inner_h, inner_i);
	float distance = distanceFromTrack(inner_f, inner_g, inner_h, inner_i, inner_fcn);
	//printf("%.3f\n", distance);
	if ((outer_f > 0 && outer_g < 0 && outer_h < 0 && outer_i>0) && 
		!(inner_f>0 && inner_g<0 && inner_h<0 && inner_i>0)) {
		state_cost += 0;
	}
	else {
		state_cost += OFF_ROAD_COST;
	}
	state_cost += distance / (ROAD_WIDTH / 2) * TRACK_ERR_COST;
	cost_list[idx] = state_cost;
	//printf("%.3f\n", cost_list[idx]);
}

__global__ void calcRolloutCost(float* input, float* output) {
	/*  Calculate the total cost of each rollout and store it into an 
		array with reduced size (reduced sum no bank conflict)*/
		// Allocate shared memory
	__shared__ float partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = input[tid];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
		//printf("%4d::%.3f\n", blockIdx.x, output[blockIdx.x]);
	}
}

__global__ void min_reduction(float* input, float* output) {
	// Allocate shared memory
	__shared__ float partial_sum[32];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = min(input[i], input[i + blockDim.x]);
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] = min(partial_sum[threadIdx.x], partial_sum[threadIdx.x + s]);
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
		//printf("result: %.3f\n", partial_sum[0]);
		//printf("%4d::%.3f\n", blockIdx.x, v_r[blockIdx.x]);
	}
}

__global__ void calcWTilde(float* w_tilde, float* rho, float* cost_list) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%.3f\n", rho[0]);
	w_tilde[idx] = __expf(-1/LAMBDA*(cost_list[idx]-rho[0]));
	//printf("%.3f %3f\n", w_tilde[idx], - 1 / LAMBDA * (cost_list[idx] - rho[0]));
}

__global__ void genW(Control* u_opt, float* eta, float* w_tilde, Control* V) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%.3f\n", eta[0]);
	__shared__ float w;
	w = w_tilde[blockIdx.x] / eta[0];
	//printf("%.3f %.3f\n", w_tilde[blockIdx.x], w);
	//printf("%.3f %.3f\n", V[idx].vd, V[idx].vd);
	u_opt[idx].vd = V[idx].vd * w;
	u_opt[idx].wd = V[idx].wd * w;
	//printf("%.3f %.3f %.3f| %.3f %.3f\n", w, eta[0], w_tilde[blockIdx.x], u_opt[idx].vd, u_opt[idx].wd);
}

__global__ void wsum_reduction(Control *input, Control* output) {
	/*  Calculate the total cost of each rollout and store it into an
		array with reduced size (reduced sum no bank conflict)*/
		// Allocate shared memory
	__shared__ Control partial_sum[1024];

	// Calculate thread ID
	int tid = threadIdx.x * gridDim.x + blockIdx.x;
	//if (tid >1000) printf("%d\n", tid);
	//if(threadIdx.x > 1000) printf("%d %d %d %d | %d\n", threadIdx.x , gridDim.x, blockDim.x, blockIdx.x, tid);
	// Load elements into shared memory
	partial_sum[threadIdx.x] = input[tid];
	//if (blockIdx.x == 0 && threadIdx.x>1000) printf("%d %d\n", threadIdx.x, tid);
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x].vd += partial_sum[threadIdx.x + s].vd;
			partial_sum[threadIdx.x].wd += partial_sum[threadIdx.x + s].wd;
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
		//printf("%4d::%.3f %.3f\n", blockIdx.x, output[blockIdx.x].vd, output[blockIdx.x].wd);
	}
}

__host__ Pos getIntersec(Track line1, Track line2) {
	/*  Calculate the intersection point of two line fcns*/
	Pos intersec;
	intersec.x = (line1.b * line2.c - line2.b * line1.c) / (line1.a * line2.b - line2.a * line1.b);
	intersec.y = (line1.c * line2.a - line2.c * line1.a) / (line1.a * line2.b - line2.a * line1.b);
	return intersec;
}

__host__ void build_track(Track* inner_fcn, Track* outer_fcn, Pos* mid_intersec, Pos* inner_intersec,
	Pos* outer_intersec, Track* mid_fcn) {
	/*  Build the boundaries of a designated path, also return 
		the intersection points of the middle path and boundaries*/
	float mid_m[4] = { -mid_fcn[0].a, -mid_fcn[1].a, -mid_fcn[2].a, -mid_fcn[3].a };
	//printf("slope: %.3f %.3f %.3f %.3f\n", mid_m[0], mid_m[1], mid_m[2], mid_m[3]);
	
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

	//printf("inner0: %.3f %.3f %.3f\n", inner_fcn[0].b, inner_fcn[0].a, inner_fcn[0].c);
	//printf("inner0: %.3f %.3f %.3f\n", inner_fcn[1].b, inner_fcn[1].a, inner_fcn[1].c);
	//printf("inner0: %.3f %.3f %.3f\n", inner_fcn[2].b, inner_fcn[2].a, inner_fcn[2].c);
	//printf("inner0: %.3f %.3f %.3f\n", inner_fcn[3].b, inner_fcn[3].a, inner_fcn[3].c);

	//printf("mid itrsc0: %.3f %.3f\n", mid_intersec[0].x, mid_intersec[0].y);
	//printf("mid itrsc0: %.3f %.3f\n", mid_intersec[1].x, mid_intersec[1].y);
	//printf("mid itrsc0: %.3f %.3f\n", mid_intersec[2].x, mid_intersec[2].y);
	//printf("mid itrsc0: %.3f %.3f\n", mid_intersec[3].x, mid_intersec[3].y);

	//printf("out itrsc0: %.3f %.3f\n", outer_intersec[0].x, outer_intersec[0].y);
	//printf("out itrsc0: %.3f %.3f\n", outer_intersec[1].x, outer_intersec[1].y);
	//printf("out itrsc0: %.3f %.3f\n", outer_intersec[2].x, outer_intersec[2].y);
	//printf("out itrsc0: %.3f %.3f\n", outer_intersec[3].x, outer_intersec[3].y);
}

int main() {
	// Record runtime 
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Setup parameters and Initialize variables
	Control* host_V, * dev_V;
	Control* host_Uopt, * dev_Uopt;
	float* dev_u0;
	float  host_u0[2] = { 2, 0 };
	Control* host_U, * dev_U;
	State host_x0 = {150.91,126.71,-0.5,2,0};
	State* dev_x0;
	float  RAND_SCALAR = 2.0;
	curandState* dev_state;

	State* dev_stateList, * host_stateList;
	float* dev_state_costList;
	float* dev_rollout_costList;
	float* dev_rho;
	float* dev_w_tilde;
	float* dev_eta;
	Control* dev_u_opt, * host_u_opt;
	Control* dev_temp_u_opt;

	// Build track
	Track host_mid_fcn[4] = { 
		{1.0000, -1.33200, 82.62978},
		{1.0000, 0.75640, -240.86623},
		{1.0000, -1.36070, -33.13473},
		{1.0000, 0.47203, -35.00739}
	};
	Track* host_in_fcn, * host_out_fcn;
	Track* dev_mid_fcn, * dev_in_fcn, * dev_out_fcn;
	Pos* host_mid_intersec, * host_in_intersec, * host_out_intersec;
	Pos* dev_mid_intersec, * dev_in_intersec, * dev_out_intersec;
	
	// Setup host memory
	host_V = (Control*)malloc(K * N_HRZ * sizeof(Control));
	host_u_opt = (Control*)malloc(N_HRZ * sizeof(Control));
	host_stateList = (State*)malloc(K*N_HRZ * sizeof(State));
	host_in_fcn = (Track*)malloc(4 * sizeof(Track));
	host_out_fcn = (Track*)malloc(4 * sizeof(Track));
	host_mid_intersec = (Pos*)malloc(4 * sizeof(Pos));
	host_in_intersec = (Pos*)malloc(4 * sizeof(Pos));
	host_out_intersec = (Pos*)malloc(4 * sizeof(Pos));
	host_U = (Control*)malloc(N_HRZ * sizeof(Control));

	// Setup device memory
	cudaMalloc((void**)&dev_u0, 2 * sizeof(float));
	cudaMalloc((void**)&dev_V, K * N_HRZ * sizeof(Control));
	cudaMalloc((void**)&dev_state, K * N_HRZ * sizeof(curandState));
	cudaMalloc((void**)&dev_stateList, K*N_HRZ * sizeof(State));
	cudaMalloc((void**)&dev_x0, sizeof(State));
	cudaMalloc((void**)&dev_state_costList, K * N_HRZ * sizeof(float));
	cudaMalloc((void**)&dev_rollout_costList, K * sizeof(float));
	cudaMalloc((void**)&dev_mid_fcn, 4 * sizeof(Track));
	cudaMalloc((void**)&dev_in_fcn, 4 * sizeof(Track));
	cudaMalloc((void**)&dev_out_fcn, 4 * sizeof(Track));
	cudaMalloc((void**)&dev_mid_intersec, 4 * sizeof(Pos));
	cudaMalloc((void**)&dev_in_intersec, 4 * sizeof(Pos));
	cudaMalloc((void**)&dev_out_intersec, 4 * sizeof(Pos));
	cudaMalloc((void**)&dev_rho, K * sizeof(float));
	cudaMalloc((void**)&dev_w_tilde, K * sizeof(float));
	cudaMalloc((void**)&dev_eta, K * sizeof(float));
	cudaMalloc((void**)&dev_u_opt, N_HRZ * sizeof(Control));
	cudaMalloc((void**)&dev_temp_u_opt, K * N_HRZ * sizeof(Control));
	cudaMalloc((void**)&dev_U, N_HRZ * sizeof(Control));

	// Setup constant memory
	build_track(host_in_fcn, host_out_fcn, host_mid_intersec, host_in_intersec, host_out_intersec, host_mid_fcn);

	//cudaMemcpyToSymbol(dev_mid_fcn, host_mid_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(dev_in_fcn, host_in_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(dev_out_fcn, host_out_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(dev_mid_intersec, host_mid_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(dev_in_intersec, host_in_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(dev_out_intersec, host_out_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_mid_fcn, host_mid_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_in_fcn, host_in_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out_fcn, host_out_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mid_intersec, host_mid_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_in_intersec, host_in_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out_intersec, host_out_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice);

	// Copy host to device
	cudaMemcpy(dev_u0, host_u0, 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x0, &host_x0, sizeof(State), cudaMemcpyHostToDevice);

	// Initialize nominal conrtrol
	for (int n = 0; n < N_HRZ; n++) {
		host_U[n].vd = host_u0[0];
		host_U[n].wd = host_u0[1];
	}
	cudaMemcpy(dev_U, host_U, N_HRZ * sizeof(Control), cudaMemcpyHostToDevice);

	// Launch kernal functions
	//cudaEventRecord(start, 0);
	
	initCurand		<< <320, 128 >> > (dev_state, 1);  // Slow! might not want it in loop?
	cudaEventRecord(start, 0);
	//genControlOld		<< <1024, 40 >> > (dev_state, dev_V, dev_u0, RAND_SCALAR);
	genControl		<< <1024,40 >> > (dev_state, dev_V, dev_U, RAND_SCALAR);  // WRONG! shoud take in a sequence of controls and not a single u
	
	genState		<< <4, 256 >> > (dev_stateList, dev_x0, dev_V);
	
	costFcn			<< <160, 256 >> > (dev_state_costList, dev_stateList, dev_out_fcn, dev_in_fcn);

	calcRolloutCost	<< <1024, 40 >> > (dev_state_costList, dev_rollout_costList);

	min_reduction	<< <32 / 2, 32 >> > (dev_rollout_costList, dev_rho);
	min_reduction	<< <1     , 32 >> > (dev_rho, dev_rho);
	
	calcWTilde		<< <32, 32 >> > (dev_w_tilde, dev_rho, dev_rollout_costList);
	
	sum_reduction	<< <32 / 2, 32 >> > (dev_w_tilde, dev_eta);
	sum_reduction	<< <1, 32 >> > (dev_eta, dev_eta);//size
	
	genW			<< <1024, 40 >> > (dev_temp_u_opt, dev_eta, dev_w_tilde, dev_V);
	wsum_reduction	<< <40, 1024 >> > (dev_temp_u_opt, dev_u_opt);
	cudaEventRecord(stop, 0);

	// Show runtime in miliseconds
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("kernal runtime: %.5f ms\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Acts as a syncing buffer so no need for additional synhronization
	cudaMemcpy(host_V, dev_V, K*N_HRZ * sizeof(Control), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_stateList, dev_stateList, K*N_HRZ * sizeof(State), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_u_opt, dev_u_opt, N_HRZ * sizeof(Control), cudaMemcpyDeviceToHost);

	printf("returned random value is %.1f %.1f %.1f\n", host_V[0].vd, host_V[1].vd, host_V[2].wd);
	printf("returned state list value is %.1f %.1f %.1f %.1f\n", host_stateList[1].x, host_stateList[1].y, host_stateList[1].phi, host_stateList[1].v0);
	printf("returned state list value is %.1f %.1f %.1f %.1f\n", host_stateList[5].x, host_stateList[5].y, host_stateList[5].phi, host_stateList[5].v0);
	for (int i = 0; i < N_HRZ-15; i++) {
		printf("%3.2f ", host_u_opt[i].vd);
	}
	printf("\n");
	for (int i = 0; i < N_HRZ-15; i++) {
		printf("%3.2f ", host_u_opt[i].wd);
	}
	printf("\n");
	//// Vector size
	//int n = 1 << 16;
	//size_t bytes = n * sizeof(int);

	//// Original vector and result vector
	//int* h_v, * h_v_r;
	//int* d_v, * d_v_r;

	//// Allocate memory
	//h_v = (int*)malloc(bytes);
	//h_v_r = (int*)malloc(bytes);
	//cudaMalloc(&d_v, bytes);
	//cudaMalloc(&d_v_r, bytes);

	//// Initialize vector
	//initialize_vector(h_v, n);

	//// Copy to device
	//cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

	//// TB Size
	//int TB_SIZE = SIZE;

	//// Grid Size (cut in half) (No padding)
	//int GRID_SIZE = (n + TB_SIZE - 1) / TB_SIZE / 2;

	//// Call kernel
	//sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);

	//sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);

	//// Copy to host;
	//cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	//// Print the result
	////printf("Accumulated result is %d \n", h_v_r[0]);
	////scanf("Press enter to continue: ");
	//printf("result: %d\n", h_v_r[0]);
	//assert(h_v_r[0] == 65536 * 2);

	//printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}