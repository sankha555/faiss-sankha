#ifndef FAISS_INDEX_COMPASS_H
#define FAISS_INDEX_COMPASS_H

#include <stdint.h>

#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/ProductQuantizer.h>

namespace faiss {
    /* Index to encapsulate all Compass functionalities */

    struct IndexCompass : IndexHNSW {
        int efn = -1;
        int efspec = -1; 

        IndexPQ* pq_index;

        IndexCompass();

        IndexCompass(
            IndexHNSW* hnsw_index,
            IndexPQ* pq_index,
            int efn,
            int efspec,
            MetricType metric = METRIC_L2
        );
        
        void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
    };


    // struct IndexCompass2 : IndexHNSW, IndexPQ {
    //     int efn = -1;
    //     int efspec = -1;

    //     IndexCompass2();

    //     IndexCompass2(
    //         IndexHNSW* hnsw_index,
    //         IndexPQ* pq_index,
    //         int efn,
    //         int efspec,
    //         MetricType metric = METRIC_L2
    //     );

    //     void search(
    //         idx_t n,
    //         const float* x,
    //         idx_t k,
    //         float* distances,
    //         idx_t* labels,
    //         const SearchParameters* params = nullptr
    //     ) const override;
    // };
}

#endif