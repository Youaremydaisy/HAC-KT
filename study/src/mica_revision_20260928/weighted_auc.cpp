#include <cstdint>
#include <limits>

// Exact tie-aware weighted Mann-Whitney AUC for pre-sorted predictions.
extern "C" double cluster_auc(std::int64_t n, const double* score,
    const std::uint8_t* y, const std::int32_t* group, const std::int32_t* weight) {
    double negative_before=0., positive_total=0., numerator=0.;
    std::int64_t i=0;
    while(i<n) {
        std::int64_t j=i; double positive=0.,negative=0.;
        do {
            const double w=weight[group[j]];
            if(y[j]) positive+=w; else negative+=w;
            ++j;
        } while(j<n && score[j]==score[i]);
        numerator+=positive*(negative_before+.5*negative);
        negative_before+=negative; positive_total+=positive; i=j;
    }
    const double den=negative_before*positive_total;
    return den>0 ? numerator/den : std::numeric_limits<double>::quiet_NaN();
}
