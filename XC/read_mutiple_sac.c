#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "sac.h"
#include "sacio.c"

#define K_LEN_8 8
#define K_LEN_16 16

#define PATH_MAX        4096	/* # chars in a path name including nul */

typedef struct
{
  pthread_mutex_t mtx;
  int valid; /* -1: default; 1: ready to file; 2: finish to file */
  char fname[PATH_MAX];
  SACHEAD *phead;
  float *pdata;
} SHAREDITEM;

// NOTE: write all ncf.sac into just one file
int write_multiple_sac(const char *filename, SHAREDITEM *pItem, int paircnt) {
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s for writing ncf.sac!\n", filename);
        return -1;
    }

    for (int i = 0; i < paircnt; i++) {
      SHAREDITEM *ptr = pItem + i;
      // 写入SAC头部信息
      if (fwrite(ptr->phead, sizeof(SACHEAD), 1, fp) != 1) {
          fprintf(stderr, "Error writing SAC header for item %d.\n", i);
          fclose(fp);
          return -1;
      }

      // 写入SAC数据
      int data_size = ptr->phead->npts * sizeof(float); // 计算数据大小
      if (fwrite(ptr->pdata, data_size, 1, fp) != 1) {
          fprintf(stderr, "Error writing SAC data for item %d.\n", i);
          fclose(fp);
          return -1;
      }
    }

    // 使用ftell获取文件大小
    long filesize = ftell(fp);
    if (filesize == -1) {
        fprintf(stderr, "Error determining file size.\n");
        fclose(fp);
        return -1;
    } else {
        double filesizeGB = filesize / (double)(1 << 30); // 转换为GB
        printf("[INFO]: File size: %.3f GB.\n", filesizeGB);
    }

    fclose(fp);
    return 0;
}

char* generate_sac_filename(const char *dir, const char *filename, char *output) {
    sprintf(output, "%s/%s", dir, filename);
    return output;
}

// 读取大文件中的数据并将其写入到多个sac文件
int read_and_writeout_multiple_sac(const char *filename, const char *outdir) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s for reading.\n", filename);
        return -1;
    }

    SACHEAD tempHead;
    char sac_filename[PATH_MAX];
    int count = 0;

    while (fread(sac_filename, sizeof(char), PATH_MAX, fp) == PATH_MAX) {
        // 去掉文件名字符串末尾的空字符
        sac_filename[strcspn(sac_filename, "\0")] = '\0';

        printf("[INFO]: File name: %s\n", sac_filename);

        if (fread(&tempHead, sizeof(SACHEAD), 1, fp) != 1) {
            fprintf(stderr, "Error reading SAC header for item %d.\n", count);
            fclose(fp);
            return -1;
        }

        float *data = (float *)malloc(tempHead.npts * sizeof(float));
        if (data == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            fclose(fp);
            return -1;
        }

        if (fread(data, sizeof(float), tempHead.npts, fp) != tempHead.npts) {
            fprintf(stderr, "Error reading SAC data for item %d.\n", count);
            free(data);
            fclose(fp);
            return -1;
        }

        char sac_fullpath[PATH_MAX];
        generate_sac_filename(outdir, sac_filename, sac_fullpath);

        write_sac(sac_fullpath, tempHead, data);

        free(data);
        count++;
    }

    fclose(fp);
    printf("[INFO]: %d SAC files have been written to %s.\n", count, outdir);
    return 0; // 成功完成
}

int main() {
    // if (argc != 3) {
    //     fprintf(stderr, "Usage: %s <input_file> <output_dir>\n", argv[0]);
    //     return -1;
    // }

    // const char *filename = argv[1];
    // const char *outdir = argv[2];

    const char *filename = "/home/woodwood/hpc/station_2/z-test/stack_test_2year_sort/ncf/AAKH-ABNH.U-U.sac";
    const char *outdir = "/home/woodwood/hpc/station_2/decode_all_sac/ncfs";

    return read_and_writeout_multiple_sac(filename, outdir);
}
