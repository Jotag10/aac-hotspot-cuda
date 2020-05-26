#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

// Returns the current system time in microseconds 
long long get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;

}

using namespace std;

#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

#define STR_SIZE	256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
#define OPEN
//#define NUM_THREAD 4

typedef float FLOAT;

void kernel_ifs(FLOAT *result, FLOAT *temp, FLOAT *power, int col, int row, FLOAT Cap_1, FLOAT Rx_1, 
				FLOAT Ry_1, FLOAT Rz_1, FLOAT amb_temp);

/* chip parameters	*/
const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width = 0.016;

/* ambient temperature, assuming no package at all	*/
const FLOAT amb_temp = 80.0;

int num_omp_threads;


__constant__ FLOAT amb_temp_dev;
//__constant__ FLOAT Ry_1_dev;   
//__constant__ FLOAT Rx_1_dev;   
//__constant__ FLOAT Rz_1_dev; 
//__constant__ FLOAT Cap_1_dev;
//__constant__ int size_dev;
#define THREADS_PER_BLOCK 256

__global__ void kernel ( FLOAT *Ry_1_dev, FLOAT *Rx_1_dev, FLOAT *Rz_1_dev, FLOAT* Cap_1_dev, int* size_dev,
        FLOAT *result_dev, FLOAT *temp_dev, FLOAT *power_dev, FLOAT* col_minus_1_dev, FLOAT* col_plus_1_dev, FLOAT *DEBUG) {
//__global__ void kernel (FLOAT *result_dev, FLOAT *temp_dev, FLOAT *power_dev, FLOAT *Cap_1_dev) {
    // FIXME assumi que #colunas=#linhas
    unsigned int column = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    /*
    if (row >= BLOCK_SIZE_R_dev && row < size-BLOCK_SIZE_R_dev
            && column > BLOCK_SIZE_C_dev && column < size-BLOCK_SIZE_C_dev) */
    
    int size = *size_dev;
    //result_dev[size*size] = 1; 
    //if (column < size*size - 1  && column > size+1) {
    /*if (column == BLOCK_SIZE && row != 0 && row != size-1) {
        result_dev[row*size+column] =temp_dev[row*size+column]+ 
             ( (*Cap_1_dev) * (power_dev[row*size+column] + 
            (temp_dev[(row+1)*size+column] + temp_dev[(row-1)*size+column] - 2.f*temp_dev[row*size+column]) * (*Ry_1_dev) + 
            (temp_dev[row*size+column+1] + col_minus_1_dev[row] - 2.f*temp_dev[row*size+column]) * (*Rx_1_dev) + 
            (amb_temp_dev - temp_dev[row*size+column]) * (*Rz_1_dev)));
    }else if (column == size - BLOCK_SIZE - 1 && row != 0 && row != size-1) {
        result_dev[row*size+column] =temp_dev[row*size+column]+ 
             ( (*Cap_1_dev) * (power_dev[row*size+column] + 
            (temp_dev[(row+1)*size+column] + temp_dev[(row-1)*size+column] - 2.f*temp_dev[row*size+column]) * (*Ry_1_dev) + 
            (col_plus_1_dev[row] + temp_dev[row*size+column-1] - 2.f*temp_dev[row*size+column]) * (*Rx_1_dev) + 
            (amb_temp_dev - temp_dev[row*size+column]) * (*Rz_1_dev)));
    } else*/ if (row < size - 15  && row > 15) {
        //*size_dev = 1023;
        //DEBUG[row*size+column] = 1.0;
        /*
        result_dev[column] =temp_dev[column]+ 
             ( (*Cap_1_dev) * (power_dev[column] + 
            (temp_dev[size+column] + temp_dev[column-size] - 2.f*temp_dev[column]) * (*Ry_1_dev) + 
            (temp_dev[column+1] + temp_dev[column-1] - 2.f*temp_dev[column]) * (*Rx_1_dev) + 
            (amb_temp_dev - temp_dev[column]) * (*Rz_1_dev)));*/
        result_dev[row*size+column] =temp_dev[row*size+column]+ 
             ( (*Cap_1_dev) * (power_dev[row*size+column] + 
            (temp_dev[(row+1)*size+column] + temp_dev[(row-1)*size+column] - 2.f*temp_dev[row*size+column]) * (*Ry_1_dev) + 
            (temp_dev[row*size+column+1] + temp_dev[row*size+column-1] - 2.f*temp_dev[row*size+column]) * (*Rx_1_dev) + 
            (amb_temp_dev - temp_dev[row*size+column]) * (*Rz_1_dev)));
            
    }

}

/* Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp, FLOAT *power, int row, int col) 
{
	#ifdef VERBOSE
	int i = 0;
	#endif
    long soma = 0;

	FLOAT grid_height = chip_height / row;
	FLOAT grid_width = chip_width / col;

	FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

	FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1=1.f/Rx;
    FLOAT Ry_1=1.f/Ry;
    FLOAT Rz_1=1.f/Rz;
    FLOAT Cap_1 = step/Cap;
	#ifdef VERBOSE
	fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
	fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
	#endif

    int DEBUG_INT;
    FLOAT *DEBBUG_HOST;
    DEBBUG_HOST = (FLOAT *)calloc (row*col,sizeof(FLOAT));
	
    
    FLOAT *col_minus_1 = (FLOAT *) calloc (col, sizeof(FLOAT));
    FLOAT *col_plus_1 = (FLOAT *) calloc (col, sizeof(FLOAT));

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // alocar memoria no gpu
    FLOAT *result_dev = NULL;
    err = cudaMalloc((void **)&result_dev, (size_t)(sizeof(FLOAT)*col*row));
    FLOAT *power_dev = NULL;
    err = cudaMalloc((void **)&power_dev, (size_t)(sizeof(FLOAT)*row*col));
    FLOAT *temp_dev = NULL;
    err = cudaMalloc((void **)&temp_dev, (size_t)(sizeof(FLOAT)*row*col));
    FLOAT *col_minus_1_dev = NULL;
    err = cudaMalloc((void **)&col_minus_1_dev, (size_t)(sizeof(FLOAT)*col));
    FLOAT *col_plus_1_dev = NULL;
    err = cudaMalloc((void **)&col_plus_1_dev, (size_t)(sizeof(FLOAT)*col));
    FLOAT *Ry_1_dev = NULL;
    err = cudaMalloc((void **)&Ry_1_dev, (size_t)sizeof(FLOAT));
    FLOAT *Rx_1_dev = NULL;
    err = cudaMalloc((void **)&Rx_1_dev, (size_t)sizeof(FLOAT));
    FLOAT *Rz_1_dev = NULL;
    err = cudaMalloc((void **)&Rz_1_dev, (size_t)sizeof(FLOAT));
    FLOAT *Cap_1_dev = NULL;
    err = cudaMalloc((void **)&Cap_1_dev, (size_t)sizeof(FLOAT));
    int *size_dev = NULL;
    err = cudaMalloc((void **)&size_dev, (size_t)sizeof(int));
    FLOAT *DEBUG = NULL;
    err = cudaMalloc((void **)&DEBUG, (size_t)(sizeof(FLOAT)*row*col));
    //transferir para o gpu
    err = cudaMemcpy(temp_dev, temp, (size_t)(sizeof(FLOAT)*col*row), cudaMemcpyHostToDevice);
    err = cudaMemcpy(power_dev, power, (size_t)(sizeof(FLOAT)*col*row), cudaMemcpyHostToDevice);
    
    err = cudaMemcpy(Ry_1_dev, &Ry_1, (size_t)sizeof(FLOAT), cudaMemcpyHostToDevice);
    err = cudaMemcpy(Rx_1_dev, &Rx_1, (size_t)sizeof(FLOAT), cudaMemcpyHostToDevice);
    err = cudaMemcpy(Rz_1_dev, &Rz_1, (size_t)sizeof(FLOAT), cudaMemcpyHostToDevice);
    err = cudaMemcpy(Cap_1_dev, &Cap_1, (size_t)sizeof(FLOAT), cudaMemcpyHostToDevice);
    err = cudaMemcpy(size_dev, &col, (size_t)sizeof(int), cudaMemcpyHostToDevice);
    //copy amb_temp to device
    cudaMemcpyToSymbol(amb_temp_dev, &amb_temp, (size_t)sizeof(FLOAT));
    //cudaMemcpyToSymbol(Ry_1_dev, &Ry_1,  (size_t)sizeof(FLOAT));
    //cudaMemcpyToSymbol(Rx_1_dev, &Rx_1,  (size_t)sizeof(FLOAT));
    //cudaMemcpyToSymbol(Rz_1_dev, &Rz_1,  (size_t)sizeof(FLOAT));
    //cudaMemcpyToSymbol(Cap_1_dev, &Cap_1, (size_t)sizeof(FLOAT));
    //cudaMemcpyToSymbol(size_dev, &col, (size_t)sizeof(FLOAT));

    dim3 blockDist(THREADS_PER_BLOCK,1,1);
    dim3 gridDist((row+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, col, 1);
    //int n_blocks = (col*row+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

    FLOAT* r = result;
    FLOAT* t = temp;
    FLOAT* tmp;
    for (int i = 0; i < num_iterations ; i++)
    {
        #ifdef VERBOSE
        fprintf(stdout, "iteration %d\n", i++);
        #endif
        result = r;

        //err = cudaMemcpyAsync(temp_dev, temp, (size_t)(sizeof(FLOAT)*col*row), cudaMemcpyHostToDevice);
        
        
        //err = cudaMemcpyAsync((temp_dev+(BLOCK_SIZE-1)*col), (temp+(BLOCK_SIZE-1)*col), (size_t)(sizeof(FLOAT)*col), cudaMemcpyHostToDevice);
        //err = cudaMemcpyAsync((temp_dev+(row-BLOCK_SIZE)*col), (temp+(row-BLOCK_SIZE)*col), (size_t)(sizeof(FLOAT)*col), cudaMemcpyHostToDevice);
        
        //for (int j = 0; j < row; j++) {
        //    err = cudaMemcpyAsync((temp_dev + j*row + BLOCK_SIZE-1), (temp + j*row+BLOCK_SIZE-1), (size_t)(sizeof(FLOAT)), cudaMemcpyHostToDevice);
        //    err = cudaMemcpyAsync((temp_dev + j*row + col-BLOCK_SIZE), (temp + j*row + col-BLOCK_SIZE), (size_t)(sizeof(FLOAT)), cudaMemcpyHostToDevice);
        //}

        for (int j = 0; j < row; j++) {
            col_minus_1[j] = *(temp + j*row+BLOCK_SIZE-1);
            col_plus_1[j] = *(temp + j*row + col-BLOCK_SIZE);
        }

        //err = cudaMemcpyAsync(col_minus_1_dev, col_minus_1, (size_t)(sizeof(FLOAT)*row), cudaMemcpyHostToDevice);
        //err = cudaMemcpyAsync(col_plus_1_dev, col_plus_1, (size_t)(sizeof(FLOAT)*row), cudaMemcpyHostToDevice);


        //kernel<<<n_blocks, THREADS_PER_BLOCK>>> (Ry_1_dev, Rx_1_dev, Rz_1_dev, 
        kernel<<<gridDist, blockDist>>> (Ry_1_dev, Rx_1_dev, Rz_1_dev, Cap_1_dev, size_dev,
                result_dev, temp_dev, power_dev, col_minus_1_dev, col_plus_1_dev, DEBUG);
        //cudaDeviceSynchronize();
        //kernel<<<n_blocks, THREADS_PER_BLOCK>>> (result_dev, temp_dev, power_dev, Cap_1_dev);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));                
            exit(EXIT_FAILURE);
        }
        //err = cudaMemcpy(&DEBUG_INT, size_dev, (size_t)(sizeof(FLOAT)), cudaMemcpyDeviceToHost);                                                            
        //printf("size - %d\n", DEBUG_INT);
        
        //err = cudaMemcpyAsync(DEBBUG_HOST, DEBUG, (size_t)(sizeof(FLOAT)*col*row), cudaMemcpyDeviceToHost);                                                            
        //for (int j = 0; j < 1024*1024; j++)
        //    if (DEBUG[j] == 1.0)
        //        soma++;
        //printf("soma:%lf\n",soma);
            //printf("DEBUG[%d] - %lf   temp[%d] - %lf\n",i, DEBBUG_HOST[i], i, temp[i]);
        //if (i == num_iterations-1)err = cudaMemcpy(result, result_dev, (size_t)(sizeof(FLOAT)*col*row), cudaMemcpyDeviceToHost);
        //if (i == num_iterations-1) 
            err = cudaMemcpy(result, result_dev, (size_t)(sizeof(FLOAT)*col*row), cudaMemcpyDeviceToHost);
        
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy vector result from device to host (error code %s)!\n", cudaGetErrorString(err));      
            exit(EXIT_FAILURE);
        }

        kernel_ifs(result, temp, power, col, row, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
        
        tmp = temp;
        temp = result;
        r = tmp;
        
        FLOAT* tmp_dev = temp_dev;
        temp_dev = result_dev;
        //result_dev = tmp_dev;
    }	

    cudaFree(result_dev);
    cudaFree(temp_dev);
    cudaFree(power_dev);
    cudaFree(Cap_1_dev);
    cudaFree(Ry_1_dev);
    cudaFree(Rx_1_dev);
    cudaFree(Rz_1_dev);
    cudaFree(size_dev);

	#ifdef VERBOSE
	fprintf(stdout, "iteration %d\n", i++);
	#endif
}

void fatal(const char *s)
{
	fprintf(stderr, "error: %s\n", s);
	exit(1);
}

void writeoutput(FLOAT *vect, int grid_rows, int grid_cols, char *file) {

    int i,j, index=0;
    FILE *fp;
    char str[STR_SIZE];

    if( (fp = fopen(file, "w" )) == 0 )
        printf( "The file was not opened\n" );


    for (i=0; i < grid_rows; i++) 
        for (j=0; j < grid_cols; j++)
        {

            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
            fputs(str,fp);
            index++;
        }

    fclose(fp);	
}

void read_input(FLOAT *vect, int grid_rows, int grid_cols, char *file)
{
  	int i, index;
	FILE *fp;
	char str[STR_SIZE];
	FLOAT val;

	fp = fopen (file, "r");
	if (!fp)
		fatal("file could not be opened for reading");

	for (i=0; i < grid_rows * grid_cols; i++) {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
			fatal("not enough lines in file");
		if ((sscanf(str, "%f", &val) != 1) )
			fatal("invalid file format");
		vect[i] = val;
	}

	fclose(fp);	
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file>\n", argv[0]);
	fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
	fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<no. of threads>   - number of threads\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
        fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char **argv)
{
	int grid_rows, grid_cols, sim_time, i;
	FLOAT *temp, *power, *result;
	char *tfile, *pfile, *ofile;
	
	/* check validity of inputs	*/
	if (argc != 8)
		usage(argc, argv);
	if ((grid_rows = atoi(argv[1])) <= 0 ||
		(grid_cols = atoi(argv[2])) <= 0 ||
		(sim_time = atoi(argv[3])) <= 0 || 
		(num_omp_threads = atoi(argv[4])) <= 0
		)
		usage(argc, argv);

	/* allocate memory for the temperature and power arrays	*/
	temp = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
	power = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
	result = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
	if(!temp || !power)
		fatal("unable to allocate memory");

	/* read initial temperatures and input power	*/
	tfile = argv[5];
	pfile = argv[6];
    ofile = argv[7];

	read_input(temp, grid_rows, grid_cols, tfile);
	read_input(power, grid_rows, grid_cols, pfile);

	printf("Start computing the transient temperature\n");
	
    long long start_time = get_time();

    compute_tran_temp(result,sim_time, temp, power, grid_rows, grid_cols);

    long long end_time = get_time();

    printf("Ending simulation\n");
    printf("Total time: %.3f seconds\n", ((float) (end_time - start_time)) / (1000*1000));

    writeoutput((1&sim_time) ? result : temp, grid_rows, grid_cols, ofile);

	/* output results	*/
#ifdef VERBOSE
	fprintf(stdout, "Final Temperatures:\n");
#endif

#ifdef OUTPUT
	for(i=0; i < grid_rows * grid_cols; i++)
	fprintf(stdout, "%d\t%g\n", i, temp[i]);
#endif
	/* cleanup	*/
	free(temp);
	free(power);

	return 0;
}



void kernel_ifs(FLOAT *result, FLOAT *temp, FLOAT *power, int col, int row, FLOAT Cap_1, FLOAT Rx_1, 
				FLOAT Ry_1, FLOAT Rz_1, FLOAT amb_temp)
{
    FLOAT delta;
    int r,c;
    int chunk;
    int num_chunk = row*col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
    int chunks_in_row = col/BLOCK_SIZE_C;
    int chunks_in_col = row/BLOCK_SIZE_R;
	
	for ( chunk = 0; chunk < num_chunk; ++chunk )
	{
		int r_start = BLOCK_SIZE_R*(chunk/chunks_in_col);
		int c_start = BLOCK_SIZE_C*(chunk%chunks_in_row); 
		int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
		int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;
	   
	   
		if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col )
		{	
			for (  r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) 
			{
                for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
                    /* Corner 1 */
                    if ( (r == 0) && (c == 0) ) {
                        delta = (Cap_1) * (power[0] +
                            (temp[1] - temp[0]) * Rx_1 +
                            (temp[col] - temp[0]) * Ry_1 +
                            (amb_temp - temp[0]) * Rz_1);
                    }	/* Corner 2 */
                    else if ((r == 0) && (c == col-1)) {
                        delta = (Cap_1) * (power[c] +
                            (temp[c-1] - temp[c]) * Rx_1 +
                            (temp[c+col] - temp[c]) * Ry_1 +
                        (   amb_temp - temp[c]) * Rz_1);
                    }	/* Corner 3 */
                    else if ((r == row-1) && (c == col-1)) {
                        delta = (Cap_1) * (power[r*col+c] + 
                            (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 + 
                            (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 + 
                        (   amb_temp - temp[r*col+c]) * Rz_1);					
                    }	/* Corner 4	*/
                    else if ((r == row-1) && (c == 0)) {
                        delta = (Cap_1) * (power[r*col] + 
                            (temp[r*col+1] - temp[r*col]) * Rx_1 + 
                            (temp[(r-1)*col] - temp[r*col]) * Ry_1 + 
                            (amb_temp - temp[r*col]) * Rz_1);
                    }	/* Edge 1 */
                    else if (r == 0) {
                        delta = (Cap_1) * (power[c] + 
                            (temp[c+1] + temp[c-1] - 2.0*temp[c]) * Rx_1 + 
                            (temp[col+c] - temp[c]) * Ry_1 + 
                            (amb_temp - temp[c]) * Rz_1);
                    }	/* Edge 2 */
                    else if (c == col-1) {
                        delta = (Cap_1) * (power[r*col+c] + 
                            (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) * Ry_1 + 
                            (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 + 
                            (amb_temp - temp[r*col+c]) * Rz_1);
                    }	/* Edge 3 */
                    else if (r == row-1) {
                        delta = (Cap_1) * (power[r*col+c] + 
                            (temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) * Rx_1 + 
                            (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 + 
                            (amb_temp - temp[r*col+c]) * Rz_1);
                    }	/* Edge 4 */
                    else if (c == 0) {
                        delta = (Cap_1) * (power[r*col] + 
                            (temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]) * Ry_1 + 
                            (temp[r*col+1] - temp[r*col]) * Rx_1 + 
                            (amb_temp - temp[r*col]) * Rz_1);
                    }
                    result[r*col+c] =temp[r*col+c]+ delta;
                }

			}
		}
	}
    	
}	

/* vim: set ts=4 sw=4  sts=4 et si ai: */
