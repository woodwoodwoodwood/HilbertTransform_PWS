#ifndef __CU_ARG_PROC_H
#define __CU_ARG_PROC_H

#define MAX_GPU_COUNT 100

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

typedef struct ARGUTYPE
{
  /* input list file of -ALST and -BLST */
  char *src_spectrum_lst;
  char *sta_spectrum_lst;
  /* output dir for CC vector */
  char *ncf_dir;
  float cclength; /* half length of output NCF */
  int gpu_id;     /* GPU ID */

  // add argument for output stack path
  char *stack_dir;
} ARGUTYPE;

void usage();
void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument);
char *createFilePath(const char *stack_dir, const char *sta_pair, const char *base_name);
char *createPwsMeanFilePath(const char *stackdir, const char *sta_pair, const char *base_name);
char *createPwsWeightedFilePath(const char *stackdir, const char *sta_pair, const char *base_name);

#endif