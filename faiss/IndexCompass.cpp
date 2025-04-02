#include <faiss/IndexCompass.h>

#include <omp.h>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>
#include <queue>
#include <random>

#include <cstdint>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

namespace faiss {

    DistanceComputer* storage_distance_computer(const Index* storage) {
        if (is_similarity_metric(storage->metric_type)) {
            return new NegativeDistanceComputer(storage->get_distance_computer());
        } else {
            return storage->get_distance_computer();
        }
    }

    IndexCompass::IndexCompass() = default;

    IndexCompass::IndexCompass(
        IndexHNSW* hnsw_index,
        IndexHNSWPQ* comp_index,
        IndexPQ* pq_index,
        int efn,
        int efspec,
        MetricType metric
    ) : IndexHNSW(*hnsw_index) {    // creates a deep copy of the HNSW index
        this->efn = efn;
        FAISS_ASSERT_MSG(this->efn > 0, "efn must be defined for Compass");

        this->efspec = efspec;
        FAISS_ASSERT_MSG(this->efspec > 0, "efspec must be defined for Compass");

        this->compressed_index = comp_index;
        this->pq_index = pq_index;
        
        float* x = (float*) malloc(sizeof(float));
        this->pq_index->pq.decode(&this->pq_index->codes.at(0), x);
        printf("x = %f\n", *x);

        printf("X = %f\n", ((FlatCodesDistanceComputer*)comp_index->get_distance_computer())->codes + ((FlatCodesDistanceComputer*)hnsw_index->get_distance_computer())->code_size);
    }

    template <class BlockResultHandler>
    void hnsw_search_with_compass(
        const IndexCompass* index,
        idx_t n,
        const float* x,
        BlockResultHandler& bres,
        const SearchParameters* params)
    {
        FAISS_THROW_IF_NOT_MSG(index->storage, "No storage index, please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");

        const HNSW& hnsw = index->hnsw;
        // const HNSW* hnsw_pq = (index->compressed_index->hnsw);

        int efSearch = hnsw.efSearch;
        if (params) {
            if (const SearchParametersHNSW* hnsw_params =
                        dynamic_cast<const SearchParametersHNSW*>(params)) {
                efSearch = hnsw_params->efSearch;
            }
        }
        size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

        idx_t check_period = InterruptCallback::get_period_hint(
                hnsw.max_level * index->d * efSearch);


        for (idx_t i0 = 0; i0 < n; i0 += check_period) {
            idx_t i1 = std::min(i0 + check_period, n);

            #pragma omp parallel if (i1 - i0 > 1)
            {
                VisitedTable vt(index->ntotal);                                 // V represented as bitvector
                typename BlockResultHandler::SingleResultHandler res(bres);

                std::unique_ptr<DistanceComputer> dis(storage_distance_computer(index->storage));
                DistanceComputer* hnswpq_dis = index->compressed_index->get_distance_computer();
                DistanceComputer* pq_dis = index->pq_index->get_distance_computer();
                // PQDistanceComputer<PQDecoder8>* pq_dis = (PQDistanceComputer<PQDecoder8>*) index->pq_index->get_distance_computer();

                #pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
                for (idx_t i = i0; i < i1; i++) {
                    res.begin(i);
                    dis->set_query(x + i * index->d);          // i-th search query SEARCH FOR: Normal HNSW query
                    hnswpq_dis->set_query(x + i * index->d);   // i-th search query SEARCH FOR: pq distance computer query
                    pq_dis->set_query(x + i * index->d);       // i-th search query SEARCH FOR: pq distance computer query

                    HNSWStats stats = hnsw.compass_search(index->compressed_index->hnsw, *dis, *hnswpq_dis, *pq_dis, res, vt, params, index->efn, index->efspec);       // Search logic for one query via Compass
                    n1 += stats.n1; 
                    n2 += stats.n2;
                    ndis += stats.ndis;
                    nhops += stats.nhops;
                    res.end();
                }
            }
            InterruptCallback::check();
        }

        hnsw_stats.combine({n1, n2, ndis, nhops});
    }

    void IndexCompass::search(
        idx_t n,                // no. of queries
        const float* x,         // queries
        idx_t k,                // get k nearest results
        float* distances,       // empty distances vectors
        idx_t* labels,          // empty labels vectors
        const SearchParameters* params)
    const {
        FAISS_THROW_IF_NOT(k > 0);

        using RH = HeapBlockResultHandler<HNSW::C>;
        RH bres(n, distances, labels, k);

        hnsw_search_with_compass(this, n, x, bres, params);

        if (is_similarity_metric(this->metric_type)) {
            // we need to revert the negated distances
            for (size_t i = 0; i < k * n; i++) {
                distances[i] = -distances[i];
            }
        }

    }


}