#include "read_segspec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>  // Include for stat()
#include <time.h>
#include <math.h>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

int readMergedSpecBuffer_onlyOneHead_mmap(const char *filename, complex *spec_buffer, size_t buffer_size) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Error opening file");
        return -1;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size");
        close(fd);
        return -1;
    }

    size_t fileSize = sb.st_size;
    void *map = mmap(NULL, fileSize, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mapping file");
        close(fd);
        return -1;
    }

    // Calculate the location of the actual data, skipping the header and the file count
    char *dataPtr = (char *)map + sizeof(SEGSPEC);
    size_t dataSectionSize = fileSize - sizeof(SEGSPEC) - sizeof(int);

    if (dataSectionSize > buffer_size * sizeof(complex)) {
        fprintf(stderr, "Buffer overflow error: not enough space in spec_buffer\n");
        munmap(map, fileSize);
        close(fd);
        return -1;
    }

    memcpy(spec_buffer, dataPtr, dataSectionSize);

    int *fileCount = (int *)(dataPtr + dataSectionSize);
    printf("Number of files merged: %d\n", *fileCount);

    printf("Total %zu complex values read_mmap from the data section.\n", dataSectionSize / sizeof(complex));

    munmap(map, fileSize);
    close(fd);
    return 0;
}

void readMergedSpecBuffer_onlyOneHead(const char *filename, complex *spec_buffer, size_t buffer_size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Unable to open %s\n", filename);
        return;
    }

    // 首先读取并忽略头区
    SEGSPEC hd;
    if (fread(&hd, sizeof(SEGSPEC), 1, file) != 1) {
        fprintf(stderr, "Error reading SEGSPEC header from %s\n", filename);
        fclose(file);
        return;
    }

    // 计算数据部分的总大小（忽略最后的整数大小）
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    long dataSectionSize = fileSize - sizeof(int) - sizeof(SEGSPEC); // 减去头区和最后的整型文件数大小

    if (dataSectionSize > buffer_size) {
        fprintf(stderr, "Buffer overflow error: not enough space in spec_buffer\n");
        fclose(file);
        return;
    }

    // 重定位到数据部分开始的位置
    fseek(file, sizeof(SEGSPEC), SEEK_SET);

    if (fread(spec_buffer, dataSectionSize, 1, file) != 1) {
        fprintf(stderr, "Error reading all data from %s\n", filename);
        fclose(file);
        return;
    }

    int fileCount;
    fseek(file, -sizeof(int), SEEK_END);
    if (fread(&fileCount, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Failed to read the number of merged files from %s\n", filename);
    } else {
        printf("Number of files merged: %d\n", fileCount);
    }

    printf("Total %zu complex values read from the data section.\n", dataSectionSize / sizeof(complex));

    fclose(file);
}

int readMergedSpec_head(const char *filename, SEGSPEC *spec_head) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Unable to open %s\n", filename);
        return -1;
    }

    // 移动到文件末尾前的int大小位置
    fseek(file, -sizeof(int), SEEK_END);
    int fileCount;
    fread(&fileCount, sizeof(int), 1, file);

    // 移动回文件开始读取SEGSPEC头信息
    rewind(file);

    // Now read the SEGSPEC header
    if (fread(spec_head, sizeof(SEGSPEC), 1, file) != 1) {
        fprintf(stderr, "Error reading SEGSPEC header from %s\n", filename);
        fclose(file);
        return -1;
    }
    fclose(file);
    return fileCount;
}

void writeFileNum(FILE *file, int fileCount) {
    fseek(file, 0, SEEK_END);  // 移动到文件末尾
    fwrite(&fileCount, sizeof(int), 1, file);  // 写入文件数
}

int mergeFiles_onlyOneHead(const char *baseDir, FILE *outputFile, int *firstFile) {
    DIR *dir;
    struct dirent *entry;
    char path[1024];
    int file_num = 0;
    SEGSPEC hd;

    dir = opendir(baseDir);
    if (dir == NULL) {
        perror("Directory not found");
        return -1;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue; // 忽略隐藏文件和目录

        snprintf(path, sizeof(path), "%s/%s", baseDir, entry->d_name);
        struct stat statbuf;
        if (stat(path, &statbuf) != 0) {
            perror("Failed to get file status");
            continue;
        }

        if (S_ISDIR(statbuf.st_mode)) {
            file_num += mergeFiles_onlyOneHead(path, outputFile, firstFile); // 递归处理子目录
        } else if (strstr(entry->d_name, ".segspec") != NULL) {
            printf("Processing file: %s\n", path);
            FILE *file = fopen(path, "rb");
            if (file) {
                if (*firstFile) {
                    // 如果是第一个文件，读取并写入头部
                    if (fread(&hd, sizeof(SEGSPEC), 1, file) == 1) {
                        fwrite(&hd, sizeof(SEGSPEC), 1, outputFile);
                    }
                    *firstFile = 0; // 标记头部已写入，后续文件不再写入头部
                } else {
                    // 跳过头部
                    fseek(file, sizeof(SEGSPEC), SEEK_SET);
                }

                // 读取并写入数据区
                char buffer[4096];
                size_t bytes;
                size_t totalBytes = 0;
                while ((bytes = fread(buffer, 1, sizeof(buffer), file)) > 0) {
                    fwrite(buffer, 1, bytes, outputFile);
                    totalBytes += bytes;
                }
                float mbWritten = totalBytes / (1024.0 * 1024.0);
                printf("Written %.2f MB to file\n", mbWritten);
                fclose(file);
                file_num++;
            } else {
                perror("Failed to open a .segspec file");
            }
        }
    }
    closedir(dir);
    return file_num;
}

void readFilesConcurrently(const char* file1, complex* buffer1, size_t size1, const char* file2, complex* buffer2, size_t size2) {
    std::thread thread1(readMergedSpecBuffer_onlyOneHead, file1, buffer1, size1);
    std::thread thread2(readMergedSpecBuffer_onlyOneHead, file2, buffer2, size2);
    
    // 等待两个线程完成
    thread1.join();
    thread2.join();

    printf("Both threads have completed\n");
}

// int main() {
//     const char *baseDir = "/home/woodwood/hpc/station_2/z-test/segspec_2year/array2";
//     const char *outputFilename = "/home/woodwood/hpc/station_2/zz-mergeFinalFile/array2_2year_merged_file.segspec";
//     FILE *outputFile = fopen(outputFilename, "w");  // 在这里打开文件
//     if (!outputFile) {
//         perror("Failed to open output file");
//         return -1;
//     }
    
//     int firstFile = 1; // 标记是否为第一个文件
//     int file_num = mergeFiles_onlyOneHead(baseDir, outputFile, &firstFile);
//     writeFileNum(outputFile, file_num);  // 写入文件数
    
//     fclose(outputFile);  // 关闭文件
//     printf("Merged %d files\n", file_num);
//     printf("All files have been merged into %s\n", outputFilename);
//     return 0;
// }

int main() {
    clock_t start, end;
    start = clock();

    SEGSPEC *spec_head_src, *spec_head_sta;
    spec_head_src = (SEGSPEC *)malloc(sizeof(SEGSPEC));
    spec_head_sta = (SEGSPEC *)malloc(sizeof(SEGSPEC));
    const char *src_mergeFile = "/home/woodwood/hpc/station_2/zz-mergeFinalFile/array1_2year_merged_file.segspec";
    const char *sta_mergeFile = "/home/woodwood/hpc/station_2/zz-mergeFinalFile/array2_2year_merged_file.segspec";
    int srccnt = readMergedSpec_head(src_mergeFile, spec_head_src);
    int stacnt = readMergedSpec_head(sta_mergeFile, spec_head_sta);
    int cclength = 500;
    int nspec = spec_head_src->nspec;
    int nstep = spec_head_src->nstep;
    float delta = spec_head_src->dt;
    int nfft = 2 * (nspec - 1);
    int nhalfcc = (int)floorf(cclength / delta);
    int ncc = 2 * nhalfcc + 1;
    // printf("stla: %f\n", spec_head->stla);
    // printf("stlo: %f\n", spec_head->stlo);
    printf("[INFO]: nspec: %d\n", nspec);
    printf("[INFO]: nstep: %d\n", nstep);
    printf("[INFO]: delta: %f\n", delta);
    printf("[INFO]: nfft: %d\n", nfft);
    printf("[INFO]: cclength: %d\n", cclength);
    printf("[INFO]: nhalfcc: %d\n", nhalfcc);
    printf("[INFO]: ncc: %d\n", ncc);
    
    printf("[INFO]: srccnt: %d\n", srccnt);

    complex *src_buffer = NULL, *src_buffer_2 = NULL;
    complex *sta_buffer = NULL, *sta_buffer_2 = NULL;
    float *ncf_buffer = NULL;

    printf("[INFO]: srccnt: %d, stacnt: %d\n", srccnt, stacnt);

    size_t total_cnt = srccnt + stacnt;
    printf("[INFO]: total_cnt: %ld\n", total_cnt);

    size_t vec_cnt = nstep * nspec;
    size_t vec_size = vec_cnt * sizeof(complex);
    printf("[INFO]: vec_cnt: %ld\n", vec_cnt);
    printf("[INFO]: vec_size: %.3f MB\n", (float)vec_size / (1024 * 1024));

    src_buffer = (complex *)malloc(vec_size * srccnt);
    sta_buffer = (complex *)malloc(vec_size * stacnt);
    src_buffer_2 = (complex *)malloc(vec_size * srccnt);
    sta_buffer_2 = (complex *)malloc(vec_size * stacnt);

    size_t src_buffer_size = vec_size * srccnt;
    size_t sta_buffer_size = vec_size * stacnt;
    printf("[INFO]: src_buffer: %.3f MB\n", (float)src_buffer_size / (1024 * 1024));
    printf("[INFO]: sta_buffer: %.3f MB\n", (float)sta_buffer_size  / (1024 * 1024));
    
    readMergedSpecBuffer_onlyOneHead_mmap(src_mergeFile, src_buffer, src_buffer_size);
    readMergedSpecBuffer_onlyOneHead_mmap(sta_mergeFile, sta_buffer, sta_buffer_size);

    // readFilesConcurrently(src_mergeFile, src_buffer_2, src_buffer_size, sta_mergeFile, sta_buffer_2, sta_buffer_size);

    // readMergedSpecBuffer_onlyOneHead("/home/woodwood/hpc/station_2/zz-mergeFinalFile/array1_merged_file.segspec", src_buffer_2, src_buffer_size);
    // readMergedSpecBuffer_onlyOneHead("/home/woodwood/hpc/station_2/zz-mergeFinalFile/array1_merged_file.segspec", sta_buffer_2, sta_buffer_size);

    end = clock();
    printf("Time taken: %.3f\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}