#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <curand_kernel.h>
#include <curand.h>

using namespace std;

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
#define LAMBDA      300.0f

#define OFF_ROAD_COST	500.0f		// Penalty for leaving the road
#define COLLISION_COST	800.0f		// Penalty for colliding into an obstacle
#define TRACK_ERR_COST	5.0f		// Penalty for tracking error (parting from the middle)

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

__global__ void sum_reduction(float* v, float* v_r) {
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

__global__ void printVar(float* var) {
	printf("====%.3f====", var[0]);
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

__device__ float distanceFromTrack(float inner_f, float inner_g, float inner_h,
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
		distance = 0;
	}

	return distance;
}

__device__ float calculateCost(State* state, Track* outer_fcn, Track* inner_fcn) {
	/*  Calculate the cost of a current state (obstacle collision part not added yet)*/

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
}

__global__ void genControl(curandState* state, Control* rand, Control* u, float scalar) {
	/*  Generate V from nominal control U and the generated pertubations E (V = U + E)
		mean 0.0 and standard deviation 1.0, scalar: stdev scalar*/

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	rand[idx].vd = curand_normal(&state[idx]) * scalar + u[threadIdx.x].vd;
	rand[idx].wd = curand_normal(&state[idx]) * scalar + u[threadIdx.x].wd;
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
		int listidx = n + (N_HRZ) * idx;
		state_list[listidx] = state_temp;
	}
	
}  

__global__ void costFcn(float* cost_list, State* state_list,  
	Track* outer_fcn, Track* inner_fcn) {
	/*  Calculate all the state costs in the prediction horizon 
		for each rollout k */

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float state_cost = 0.0f;
	State state = state_list[idx];

	float outer_f = outer_fcn[0].b * state.y + outer_fcn[0].a * state.x + outer_fcn[0].c;  
	float outer_g = outer_fcn[1].b * state.y + outer_fcn[1].a * state.x + outer_fcn[1].c;
	float outer_h = outer_fcn[2].b * state.y + outer_fcn[2].a * state.x + outer_fcn[2].c;
	float outer_i = outer_fcn[3].b * state.y + outer_fcn[3].a * state.x + outer_fcn[3].c;

	float inner_f = inner_fcn[0].b * state.y + inner_fcn[0].a * state.x + inner_fcn[0].c;
	float inner_g = inner_fcn[1].b * state.y + inner_fcn[1].a * state.x + inner_fcn[1].c;
	float inner_h = inner_fcn[2].b * state.y + inner_fcn[2].a * state.x + inner_fcn[2].c;
	float inner_i = inner_fcn[3].b * state.y + inner_fcn[3].a * state.x + inner_fcn[3].c;
	float distance = distanceFromTrack(inner_f, inner_g, inner_h, inner_i, inner_fcn);
	if ((outer_f > 0 && outer_g < 0 && outer_h < 0 && outer_i>0) && 
		!(inner_f>0 && inner_g<0 && inner_h<0 && inner_i>0)) {
		state_cost += 0;
	}
	else {
		state_cost += OFF_ROAD_COST;
	}
	state_cost += distance / (ROAD_WIDTH / 2) * TRACK_ERR_COST;
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
	__shared__ float partial_sum[40];

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
__global__ void line() {
	printf("=====================\n");
}

__global__ void printfloat(float* f) {
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	printf("float: %.3f\n", f[i]);
}

__global__ void min_reduction(float* input, float* output) {
	/*  Find the rollout with the minimum cost using sum reduction methods, 
		but changing the "sum" part to "min"*/

	__shared__ float partial_sum[K/32];
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

	for (int s = blockDim.x /4; s > 0; s >>= 1) {
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
	w_tilde[idx] = __expf(-1/LAMBDA*(cost_list[idx]-rho[0]));
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

__global__ void wsum_reduction(Control *input, Control* output) {
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

int main() {
	// Record runtime 
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Setup parameters and Initialize variables
	Control* host_V, * dev_V;
	Control output_u0;
	float  host_u0[2] = { 2, 0 };
	Control* host_U, * dev_U;
	/*State host_x0 = {150.91,126.71,-1,2};*/
	State host_x0 = { 136, 138,-1,2 };
	//State host_x0 = { 153.843, 120.072, -1.520, 2.726 };
	State* dev_x0;
	float  RAND_SCALAR = 2.0;
	curandState* dev_cstate;

	State* dev_stateList, * host_stateList;
	float* dev_state_costList;
	float* dev_rollout_costList;
	float* host_rollout_costList;
	float* dev_rho;
	float* dev_w_tilde;
	float* dev_eta;
	Control* dev_u_opt, * dev_u_opt_part, * host_u_opt;
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
	//host_U = (Control*)malloc(N_HRZ * sizeof(Control));

	// Setup device memory
	cudaMalloc((void**)&dev_cstate, int(K * N_HRZ) * sizeof(curandState));
	cudaMalloc((void**)&dev_V, int(K * N_HRZ) * sizeof(Control));
	cudaMalloc((void**)&dev_x0, sizeof(State));
	cudaMalloc((void**)&dev_stateList, int(K * N_HRZ) * sizeof(State));
	cudaMalloc((void**)&dev_state_costList, int(K * N_HRZ) * sizeof(float));
	cudaMalloc((void**)&dev_rollout_costList, int(K) * sizeof(float));
	cudaMalloc((void**)&dev_mid_fcn, 4 * sizeof(Track));
	cudaMalloc((void**)&dev_in_fcn, 4 * sizeof(Track));
	cudaMalloc((void**)&dev_out_fcn, 4 * sizeof(Track));
	cudaMalloc((void**)&dev_mid_intersec, 4 * sizeof(Pos));
	cudaMalloc((void**)&dev_in_intersec, 4 * sizeof(Pos));
	cudaMalloc((void**)&dev_out_intersec, 4 * sizeof(Pos));
	cudaMalloc((void**)&dev_rho, int(K) * sizeof(float));
	cudaMalloc((void**)&dev_w_tilde, int(K) * sizeof(float));
	cudaMalloc((void**)&dev_eta, int(K) * sizeof(float));
	cudaMalloc((void**)&dev_u_opt, int(N_HRZ) * sizeof(Control));
	cudaMalloc((void**)&dev_u_opt_part, int(N_HRZ) * sizeof(Control));
	cudaMalloc((void**)&dev_temp_u_opt, int(K * N_HRZ) * sizeof(Control));
	cudaMalloc((void**)&dev_U, int(N_HRZ) * sizeof(Control));

	// Setup constant memory
	build_track(host_in_fcn, host_out_fcn, host_mid_intersec, host_in_intersec, host_out_intersec, host_mid_fcn);
	cudaMemcpy(dev_mid_fcn, host_mid_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_in_fcn, host_in_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out_fcn, host_out_fcn, 4 * sizeof(Track), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mid_intersec, host_mid_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_in_intersec, host_in_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out_intersec, host_out_intersec, 4 * sizeof(Pos), cudaMemcpyHostToDevice);

	// Initialize nominal conrtrol
	for (int n = 0; n < N_HRZ; n++) {
		host_U[n].vd = host_u0[0];
		host_U[n].wd = host_u0[1];
	}
	
	std::fstream outputFile;
	std::fstream testFile;
	std::fstream cost;
	std::fstream predFile;
	// create a name for the file output
	outputFile.open("MPC_output4.csv", ios::out | ios::app);
	testFile.open("testFile.csv", ios::out | ios::app);
	cost.open("costroll.csv", ios::out | ios::app);
	predFile.open("predstate.csv", ios::out | ios::app);
	initCurand << <N_HRZ * 8, K / 8 >> > (dev_cstate, 0);

	for (double it = 0; it < 30; it++) {

		cudaMemcpy(dev_U, host_U, int(N_HRZ) * sizeof(Control), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_x0, &host_x0, sizeof(State), cudaMemcpyHostToDevice);

		// Launch kernal functions
		//cudaEventRecord(start, 0);

		/*initCurand << <N_HRZ * 8, K / 8 >> > (dev_cstate, 0);*/// (long)it);  // Slow! might not want it in loop?
		//cudaEventRecord(start, 0);
		//printControl << <1, 1 >> > (dev_U);
		genControl << <K, N_HRZ >> > (dev_cstate, dev_V, dev_U, RAND_SCALAR);  // WRONG! shoud take in a sequence of controls and not a single u

		genState << <K / 256, K / 4 >> > (dev_stateList, dev_x0, dev_V);

		costFcn << <N_HRZ * 4, K / 4 >> > (dev_state_costList, dev_stateList, dev_out_fcn, dev_in_fcn);

		calcRolloutCost2 << <K, N_HRZ >> > (dev_state_costList, dev_rollout_costList);

		min_reduction << <K / 32 / 2, K / 32 >> > (dev_rollout_costList, dev_rho);
		//line << <1, 1 >> > ();
		//printfloat << <1, 1 >> > (dev_rho);
		min_reduction2 << <1, K / 32 >> > (dev_rho, dev_rho);
		printMinCost << <1, 1 >> > (dev_rho);
		calcWTilde << <K / 32, K / 32 >> > (dev_w_tilde, dev_rho, dev_rollout_costList);

		sum_reduction << <K / 32 / 2, K / 32 >> > (dev_w_tilde, dev_eta);
		//line << <1, 1 >> > ();
		sum_reduction << <1, K / 32 >> > (dev_eta, dev_eta);//size

		//printfloat << <1, 1 >> > (dev_eta);
		genW << <K, int(N_HRZ) >> > (dev_temp_u_opt, dev_eta, dev_w_tilde, dev_V);
		wsum_reduction << <N_HRZ, K >> > (dev_temp_u_opt, dev_u_opt);
		//line << <1, 1 >> > ();
		// If 1024 threads are too much for the hardware, the method below can be adapted

		//wsum_reduction_partial << <40, 512 >> > (dev_temp_u_opt, dev_u_opt, 0);
		//wsum_reduction_partial << <40, 512 >> > (dev_temp_u_opt, dev_u_opt_part, 512);
		//sumControl << <20, 20 >> > (dev_u_opt_part, dev_u_opt);

		//cudaEventRecord(stop, 0);

		//// Show runtime in miliseconds
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&time, start, stop);
		////printf("kernal runtime: %.5f ms\n", time);
		//cudaEventDestroy(start);
		//cudaEventDestroy(stop);

		// Acts as a syncing buffer so no need for additional synhronization
		cudaMemcpy(host_V, dev_V, int(K * N_HRZ) * sizeof(Control), cudaMemcpyDeviceToHost);
		cudaMemcpy(host_stateList, dev_stateList, int(K * N_HRZ) * sizeof(State), cudaMemcpyDeviceToHost);
		cudaMemcpy(host_u_opt, dev_u_opt, int(N_HRZ) * sizeof(Control), cudaMemcpyDeviceToHost);
		cudaMemcpy(host_rollout_costList, dev_rollout_costList, int(K) * sizeof(float), cudaMemcpyDeviceToHost);
		
		
		for (int j = 0; j < 40 * 1024; j++) {
			testFile << host_stateList[j].x << ",";
		}
		testFile << std::endl;
		for (int j = 0; j < 40 * 1024; j++) {
			testFile << host_stateList[j].y << ",";
		}
		testFile << std::endl;
		for (int j = 0; j < 40 * 1024; j++) {
			testFile << host_stateList[j].phi << ",";
		}
		testFile << std::endl;
		for (int j = 0; j < 40 * 1024; j++) {
			testFile << host_stateList[j].v0 << ",";
		}
		testFile << std::endl;

		for (int j = 0; j < 1024; j++) {
			cost << host_rollout_costList[j] << std::endl;
		}

		State predstate[40];
		predstate[0] = host_x0;
		for (int i = 0; i < N_HRZ; i++) {
			int count = i - 1;
			if (count < 0) count = 0;
			predstate[i] = bicycleModel(predstate[count], host_u_opt[i]);
		}

		for (int j = 0; j < 40 ; j++) {
			predFile << predstate[j].x << ",";
		}
		predFile << std::endl;
		for (int j = 0; j < 40 ; j++) {
			predFile << predstate[j].y << ",";
		}
		predFile << std::endl;
		for (int j = 0; j < 40 ; j++) {
			predFile << predstate[j].phi << ",";
		}
		predFile << std::endl;
		for (int j = 0; j < 40 ; j++) {
			predFile << predstate[j].v0 << ",";
		}
		predFile << std::endl;
		for (int j = 0; j < 40; j++) {
			predFile << host_u_opt[j].vd << ",";
		}
		predFile << std::endl;
		for (int j = 0; j < 40; j++) {
			predFile << host_u_opt[j].wd << ",";
		}
		predFile << std::endl;
		

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

		// Might want to write the rollout states to a csv file?


		// Generate the predicted next state with u_opt
		//printf("orig: %.3f %.3f\n", host_x0.x, host_x0.y);
		host_x0 = bicycleModel(host_x0, host_u_opt[0]);
		printf("%.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", host_x0.x, host_x0.y, host_x0.phi, host_x0.v0, host_u_opt[0].vd, host_u_opt[0].wd);
		//outputFile << it << "," << host_x0.x << "," << host_x0.y << "," << host_x0.phi << "," << host_x0.v0 << "," << host_u_opt[0].vd << "," << host_u_opt[0].wd << std::endl;
		//printf("%.3f, %.3f, %.3f, %.3f\n", host_x0.x, host_x0.y, host_x0.phi, host_x0.v0);
		// Shift the control by 1 

		output_u0 = host_u_opt[0];
		memmove(host_u_opt, &host_u_opt[1], int(N_HRZ - 1) * sizeof(Control)); // use memmove instead of memcpy because the destination overlaps the source
		host_u_opt[int(N_HRZ)] = host_u_opt[int(N_HRZ) - 1];
		memcpy(host_U, host_u_opt, int(N_HRZ) * sizeof(Control));
	}
	outputFile.close();
	testFile.close();
	cost.close();
	return 0;
}