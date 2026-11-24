#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Diagonal assumed-density filtering for a logistic response model. This is
// an approximate Bayesian state update, not an exact posterior or new filter.
struct Belief {
    double mean[3] = {0., 0., 0.};
    double variance[3] = {1., 1., 1.};
    double last_innovation = 0.;
};
static void emit(const Belief& s, float* out) {
    for (int j=0;j<3;++j) {out[j]=s.mean[j];out[3+j]=s.variance[j];}
    out[6]=s.last_innovation;
}
static void observe(Belief& s,double y,double b) {
    const double process[3]={.001,.02,.1};
    for (int j=0;j<3;++j) {
        const double prior=std::min(4.,s.variance[j]+process[j]);
        const double p=1./(1.+std::exp(-std::clamp(s.mean[j]-b,-25.,25.)));
        const double fisher=p*(1.-p);
        s.variance[j]=1./(1./prior+fisher);
        if (j==1) s.last_innovation=(y-p)/std::sqrt(std::max(.01,fisher+prior*fisher*fisher));
        s.mean[j]=std::clamp(s.mean[j]+s.variance[j]*(y-p),-8.,8.);
    }
}
extern "C" int mica_beliefs(int64_t students,const int64_t* offsets,const int32_t* skills,
        const int32_t* items,const float* answers,const float* difficulty,
        int32_t skill_count,int32_t item_count,float* output) {
    std::vector<Belief> sk(skill_count),it(item_count);
    std::vector<int64_t> skstamp(skill_count,-1),itstamp(item_count,-1);
    for (int64_t u=0;u<students;++u) {
        Belief global;
        for (int64_t t=offsets[u];t<offsets[u+1];++t) {
            int32_t k=skills[t],q=items[t];
            if(k<0 || k>=skill_count || q<0 || q>=item_count)return 1;
            float* row=output+t*21;
            emit(global,row);
            if(k>0){if(skstamp[k]!=u){sk[k]=Belief();skstamp[k]=u;}emit(sk[k],row+7);}
            if(q>0){if(itstamp[q]!=u){it[q]=Belief();itstamp[q]=u;}emit(it[q],row+14);}
            observe(global,answers[t],difficulty[t]);
            if(k>0)observe(sk[k],answers[t],difficulty[t]);
            if(q>0)observe(it[q],answers[t],difficulty[t]);
        }
    }
    return 0;
}
