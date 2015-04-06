#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>

#include "util.h"
#define KERNEL_FILE "/home/sse/opencl_study/OpenCL_CPHD/sum/sum.cl"

#define N 15*10*1024
#define NUM_SUM 10000


double fRand(double fmin,double fmax)
{
	double f = (double)rand()/RAND_MAX;
	return (double) fmin + f*(fmax-fmin);
}

void GenerateInput(double* A,double* B)
{
	int i;
	for (int i = 0; i < N; ++i)
	{
		  A[i] = fRand(-15900.35, 19870.59);
          B[i] = fRand(-15400.65, 15480.68);
	}
}


double mysecond() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

cl_platform_id          platform_id[100];
cl_device_id            device_id[100];
cl_context              context;
cl_command_queue        command_queue;
cl_int ret;

void getDevices(cl_device_type deviceType)
{
    cl_uint         platforms_n = 0;
    cl_uint         devices_n   = 0;

    clGetPlatformIDs(100, platform_id, &platforms_n);
    clGetDeviceIDs(platform_id[0], deviceType, 100, device_id, &devices_n);

    // Create an OpenCL context.
    context = clCreateContext(NULL, devices_n, device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("\nError at clCreateContext! Error code %i\n\n", ret);
        exit(1);
    }

    // Create a command queue.
    command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
        exit(1);
    }
}
int main(int argc, char const *argv[])
{
	/* code */
	cl_program clProgram;
	cl_kernel clKernel;
	cl_int errcode;
	//Initialize opencl
	getDevices(CL_DEVICE_TYPE_ALL);
    int size = N;
    //Allocate host memory
    double *A = ( double* ) malloc( size * sizeof( double ) );
    double *B = ( double* ) malloc( size * sizeof( double ) );

    // Initialize values
    GenerateInput(A, B);

    // OpenCL device memory
    cl_mem d_A;
    cl_mem d_B;
    cl_mem d_num_errors;

     // Setup device memory
    unsigned int mem_size = sizeof(double) * size;
    d_num_errors = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE,
                                  sizeof(int), NULL, &errcode);
    CL_CHECK_ERROR(errcode);
    d_A = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         mem_size, A, &errcode);
    CL_CHECK_ERROR(errcode);
    d_B = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         mem_size, B, &errcode);
    CL_CHECK_ERROR(errcode);

    /*
        CL_MEM_READ_WRITE	    This flag specifies that the memory object will be read and written by a kernel. This is the default.
     --------------------------------------------------------------------------------------------------------------------------------
       CL_MEM_WRITE_ONLY	    This flags specifies that the memory object will be written but not read by a kernel.
	 --------------------------------------------------------------------------------------------------------------------------------
      CL_MEM_READ_ONLY	        This flag specifies that the memory object is a read-only memory object when used inside a kernel.
	 --------------------------------------------------------------------------------------------------------------------------------
     CL_MEM_USE_HOST_PTR	   This flag is valid only if host_ptr is not NULL. If specified, it indicates that the application wants the OpenCL implementation 
     						    to use memory referenced by host_ptr as the storage bits for the memory object.
	 --------------------------------------------------------------------------------------------------------------------------------
    CL_MEM_ALLOC_HOST_PTR	   This flag specifies that the application wants the OpenCL implementation to allocate memory from host accessible memory.
     --------------------------------------------------------------------------------------------------------------------------------
     CL_MEM_COPY_HOST_PTR	   This flag is valid only if host_ptr is not NULL. If specified, it indicates that the application wants the OpenCL implementation 
                               to allocate memory for the memory object and copy the data from memory referenced by host_ptr.
 
    */

    //load and build opencl kernel
    FILE* theFile = fopen(KERNEL_FILE,"r");
    if (!theFile)
    {
    	fprintf(stderr, "Failed to load kernel file.\n");
        exit(1);
    }

    char* source_str;

    // Obtain length of source file.
    fseek(theFile, 0, SEEK_END);
    size_t source_size = ftell(theFile);
    rewind(theFile);

    // Read in the file.
    source_str = (char*) malloc(sizeof(char) * (source_size+2));
    fread(source_str, 1, source_size, theFile);
    fclose(theFile);
    source_str[source_size] = '\0';
    CL_CHECK_ERROR(errcode);

    clProgram = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &errcode);

    CL_CHECK_ERROR(errcode);

    free(source_str);

    errcode = clBuildProgram(clProgram, 1,
                             &device_id[0], NULL, NULL, NULL);
    CL_CHECK_ERROR(errcode);
    
    cl_build_status status;
    size_t logSize;
    char* programLog;
    clGetProgramBuildInfo(clProgram, device_id[0], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);
    // check build log
    clGetProgramBuildInfo(clProgram, device_id[0],
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    programLog = (char*) calloc (logSize+1, sizeof(char));
    clGetProgramBuildInfo(clProgram, device_id[0],
                          CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
    printf("Build; error=%d, status=%d, programLog:\n\n%s", errcode, status, programLog);
    free(programLog);
    clKernel = clCreateKernel(clProgram, "simpleSum", &errcode);
    CL_CHECK_ERROR(errcode);

    size_t localWorkSize[2],globalWorkSize[2];
    int kernel_error = 0;
    clEnqueueWriteBuffer(command_queue, d_num_errors, CL_FALSE, 0, sizeof(int), &kernel_error, 0, NULL, NULL);
    clFinish(command_queue);
    int n = NUM_SUM;
    errcode |= clSetKernelArg(clKernel, 0,
                              sizeof(cl_mem), (void *)&d_A);
    errcode |= clSetKernelArg(clKernel, 1,
                              sizeof(cl_mem), (void *)&d_B);
    errcode |= clSetKernelArg(clKernel, 2,
                              sizeof(cl_mem), (void *)&d_num_errors);
    errcode |= clSetKernelArg(clKernel, 3,
                              sizeof(int), (void *)&n);
    CL_CHECK_ERROR(errcode);

    CL_CHECK_ERROR(errcode);

     localWorkSize[0] = 256;
    globalWorkSize[0] = N;

    clFinish(command_queue);

    // Run kernel
    double timeG = mysecond();
    errcode = clEnqueueNDRangeKernel(command_queue,
                                     clKernel, 1, NULL, globalWorkSize,
                                     localWorkSize, 0, NULL, NULL);
    CL_CHECK_ERROR(errcode);
    clFinish(command_queue);
    timeG = mysecond() - timeG;

    int* kernel_errors = (int*)malloc(sizeof(int));
    errcode = clEnqueueReadBuffer(command_queue,
                                  d_num_errors, CL_TRUE, 0, sizeof(int),
                                   kernel_errors, 0, NULL, NULL);
    CL_CHECK_ERROR(errcode);

    printf("check kernel errors = %d\n",*kernel_errors);
    printf("kernel time: %f\n", timeG);




    // Clean up memory
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_num_errors);

    clReleaseContext(context);
    clReleaseKernel(clKernel);
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(command_queue);

	return 0;
}
