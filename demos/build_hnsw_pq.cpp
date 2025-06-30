/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>

#include <sys/time.h>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#define DC(classname) classname* ix = dynamic_cast<classname*>(index)

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);
    // n = 1;

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double mrr_metric(size_t nq, size_t max_res, size_t k, faiss::idx_t* I, faiss::idx_t* gt) {
    // computes Mean Reciprocal Rank @ k (MRR@k metric)
    // I is the result matrix of size [nq x max_res]
    // gt is the ground-truth matrix of size [nq x max_res]
    double mrr = 0;
    for(int q = 0; q < nq; q++){
        for(int rank = 0; rank < k; rank++){
            if(I[q*max_res + rank] == gt[q*max_res]){
                mrr += 1.0/(rank+1);
                break;
            }
        }
        // if the entire for loop is traversed without a match, RR of 0 is counted
        // since the first correct result was not found in the first k search results
    }
    return mrr/nq;
}

int main() {
    double t0 = elapsed();

    const char* index_key = "HNSW";
    faiss::IndexHNSW* hnsw_index;
    faiss::IndexPQ* pq_index;
    size_t d = 128;                          // dimensions
    size_t M = 64;                     // max. degree 
    size_t q = 8;                      // no. of subvectors for PQ   
    bool use_compressed = true;

    size_t efSearch = 64;
    size_t efConstruction = 80;


    bool index_in_file = false;
    const char* hnsw_path = "./index/sift100m_hnsw_64_80_2_ip.index";
    const char* index_pq_path = "./index/sift100m_index_64_80_2_ip.index";
    const char* pq_path = "./index/sift100m_pq_full_32_ip.index";

    {
        size_t nt;
        d = 128;

        printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
            elapsed() - t0,
            index_key,
            d);

        if(index_in_file){
            // read index from file
            hnsw_index = (faiss::IndexHNSWFlat*) faiss::read_index(hnsw_path);
        }else{
            // Create a new index from scratch
            printf("[%.3f s] Loading train set\n", elapsed() - t0);
            float* xt = fvecs_read("./sift10m/sift_learn.fvecs", &d, &nt);

            
            hnsw_index = new faiss::IndexHNSWFlat(d, M, faiss::METRIC_L2); 
            hnsw_index->hnsw.efSearch = efSearch;
            hnsw_index->hnsw.efConstruction = efConstruction; 

            pq_index = new faiss::IndexPQ(d, q, 8);
            
            printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);
            hnsw_index->train(nt, xt);

            pq_index->train(nt, xt);

            delete[] xt;
        }
        
    }


    {
        if(!index_in_file){
            printf("[%.3f s] Loading database\n", elapsed() - t0);

            size_t nb, d2;
            float* xb = fvecs_read("./sift100m/sift100m_base.fvecs", &d2, &nb);
            assert(d == d2 || !"dataset does not have same dimension as train set");

            printf("[%.3f s] Indexing database, size %ld*%ld\n",
                elapsed() - t0,
                nb,
                d);
            hnsw_index->add(nb, xb);
            write_index(hnsw_index, hnsw_path);

            pq_index->add(nb, xb);

            // faiss::ProductQuantizer* pq = &pq_index->pq;    
            // std::cout << pq->M << "\n";
            // write_ProductQuantizer(pq, pq_path);
            
            // write pq codes
            write_index(pq_index, pq_path);

            delete[] xb;
        } else {
            faiss::ProductQuantizer* pq = &pq_index->pq;
            std::cout << pq->M << "\n";
        }

    }


    size_t nq;
    float* xq;

    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read("./sift10m/sift_query.fvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    size_t k;         // no. of results per query in the GT
    faiss::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
            elapsed() - t0,
            nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int* gt_int = ivecs_read("./sift10m/sift_groundtruth.ivecs", &k, &nq2);
        assert(nq2 == nq || !"incorrect no. of ground truth entries");

        gt = new faiss::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }
   

    { 
        printf("[%.3f s] Performing search on %ld queries\n",
            elapsed() - t0,
            nq);

        // output buffers
        faiss::idx_t* I = new faiss::idx_t[nq * k];
        float* D = new float[nq * k];

        // index->search(nq, xq, k, D, I);
        hnsw_index->search(nq, xq, k, D, I);

        printf("[%.3f s] Computing search-accuracy\n", elapsed() - t0);

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (int i = 0; i < nq; i++) {
            int gt_nn = gt[i * k];
            for (int j = 0; j < k; j++) {
                if (I[i * k + j] == gt_nn) {
                    if (j < 1)
                        n_1++;
                    if (j < 10)
                        n_10++;
                    if (j < 100)
                        n_100++;
                }
            }
        }
        // printf("R@1 = %.4f\n", n_1 / float(nq));
        printf("R@10 = %.4f\n", n_10 / float(nq));
        // printf("R@100 = %.4f\n", n_100 / float(nq));
        printf("MRR@10 = %.4f\n", mrr_metric(nq, k, 10, I, gt));

        delete[] I;
        delete[] D;
    }

    delete[] xq;
    delete[] gt;
    delete hnsw_index;
    return 0;
}
