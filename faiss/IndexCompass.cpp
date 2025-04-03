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
#include <sys/stat.h>


#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

float* fvecs_read_1(const char* fname, size_t* d_out, size_t* n_out) {
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
int* ivecs_read_1(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read_1(fname, d_out, n_out);
}

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
        IndexPQ* pq_index,
        int efn,
        int efspec,
        MetricType metric
    ) : IndexHNSW(*hnsw_index) {    // creates a deep copy of the HNSW index
        this->efn = efn;
        FAISS_ASSERT_MSG(this->efn > 0, "efn must be defined for Compass");

        this->efspec = efspec;
        FAISS_ASSERT_MSG(this->efspec > 0, "efspec must be defined for Compass");

        this->pq_index = pq_index;
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
                DistanceComputer* pq_dis = index->pq_index->get_distance_computer();

                #pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
                for (idx_t i = i0; i < i1; i++) {
                    res.begin(i);
                    dis->set_query(x + i * index->d);          // i-th search query SEARCH FOR: Normal HNSW query
                    pq_dis->set_query(x + i * index->d);       // i-th search query SEARCH FOR: pq distance computer query

                    HNSWStats stats = hnsw.compass_search(*dis, *pq_dis, res, vt, params, index->efn, index->efspec);       // Search logic for one query via Compass
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