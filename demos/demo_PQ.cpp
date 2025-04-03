
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

using namespace faiss;


const char* index_key = "HNSW";
size_t d = 128;                          // dimensions
size_t M = 32;                     // max. degree 
size_t q = 8;                      // no. of subvectors for PQ   
size_t nbits = 8;


bool use_compressed = true;
bool index_in_file = true;
const char* index_file_path = "./sift1M_index_HNSWPQ.faissindex";

size_t efSearch = 32;
size_t efConstruction = 40;

size_t nt;
size_t nb, d2;
size_t nq;
size_t k;         
size_t nq2;

float* xt;
float* xb;
float* xq;
faiss::idx_t* gt;
faiss::idx_t* I; 
float* D;

double t0 = elapsed();


void index_ops(faiss::IndexPQ* index){

    faiss::ProductQuantizer pq(index->pq);

    printf("d = %ld, q = %ld, nbits = %ld\n", pq.d, pq.M, pq.nbits);
    

}

void dock_hnsw_into_hnswpq(){
    IndexHNSWFlat* index_hnsw = (IndexHNSWFlat*) read_index("./sift1M_index_HNSWFlat.faissindex");
    IndexHNSWPQ* index_hnswpq = (IndexHNSWPQ*) read_index("./sift1M_index_HNSWPQ.faissindex");    

    // index_hnswpq->search(nq, xq, k, D, I);     
    // printf("MRR@10 = %.4f\n", mrr_metric(nq, k, 10, I, gt));

    // dock
    index_hnsw->hnsw = index_hnswpq->hnsw;
    
    index_hnsw->search(nq, xq, k, D, I);     
    printf("MRR@10 after Docking = %.4f\n", mrr_metric(nq, k, 10, I, gt));
}



int main(){

    faiss::IndexPQ* index_pq;
    faiss::IndexHNSWFlat* index_hnsw;
    faiss::IndexHNSWPQ* index_hnswpq;

    // if(index_in_file){
    //     index = (faiss::IndexPQ*) faiss::read_index(index_file_path);
    // } else {
    //     index = new faiss::IndexPQ(d, q, nbits);
    // }
    
    // if(index_in_file){
    //     index_hnsw = (faiss::IndexHNSWFlat*) faiss::read_index(index_file_path);
    // } else {
    //     index_hnsw = new faiss::IndexHNSWFlat(d, M);
    // }

    if(index_in_file){
        index_hnswpq = (faiss::IndexHNSWPQ*) faiss::read_index(index_file_path);
    } else {
        index_hnswpq = new faiss::IndexHNSWPQ(d, q, M, 8);
    }

    {   /* Dataset loading operations */
        xt = fvecs_read("/mnt/sift1M/sift_learn.fvecs", &d, &nt);
        printf("[%.3f s] Read %ld training data vectors\n", elapsed() - t0, nt);

        xb = fvecs_read("/mnt/sift1M/sift_base.fvecs", &d2, &nb);
        printf("[%.3f s] Read %ld indexing data vectors\n", elapsed() - t0, nb);

        xq = fvecs_read("/mnt/sift1M/sift_query.fvecs", &d2, &nq);
        printf("[%.3f s] Read %ld query data vectors\n", elapsed() - t0, nq);

        int* gt_int = ivecs_read("/mnt/sift1M/sift_groundtruth.ivecs", &k, &nq2);
        gt = new faiss::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }

        I = new faiss::idx_t[nq * k];
        D = new float[nq * k];


        // delete[] gt;
        // delete[] I;
        // delete[] D;
    }

    
    // if(!index_in_file){
    //     index->train(nt, xt);                       delete[] xt;
    //     printf("[%.3f s] Training completed\n", elapsed() - t0);
    
    //     index->add(nb, xb);                         delete[] xb;
    //     printf("[%.3f s] Indexing completed\n", elapsed() - t0);

    //     faiss::write_index(index, index_file_path);
    // }


    // if(!index_in_file){
    //     index_hnsw->train(nt, xt);                       delete[] xt;
    //     printf("[%.3f s] Training completed\n", elapsed() - t0);
    
    //     index_hnsw->add(nb, xb);                         delete[] xb;
    //     printf("[%.3f s] Indexing completed\n", elapsed() - t0);

    //     faiss::write_index(index_hnsw, index_file_path);
    // }


    // if(!index_in_file){
    //     index_hnswpq->train(nt, xt);                       delete[] xt;
    //     printf("[%.3f s] Training completed\n", elapsed() - t0);
    
    //     index_hnswpq->add(nb, xb);                         delete[] xb;
    //     printf("[%.3f s] Indexing completed\n", elapsed() - t0);

    //     faiss::write_index(index_hnswpq, index_file_path);
    // }

    // index_ops(index);
    // index_ops(index_hnsw);

    // index_hnswpq->search(nq, xq, k, D, I);           delete[] xq;
    // printf("[%.3f s] Search completed\n", elapsed() - t0);


    // printf("MRR@10 = %.4f\n", mrr_metric(nq, k, 10, I, gt));

    dock_hnsw_into_hnswpq();

    return 0;
}

