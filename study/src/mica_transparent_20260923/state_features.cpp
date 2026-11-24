#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

struct State {
    double theta[3] = {0., 0., 0.};
    double success[3] = {0., 0., 0.};
    double expected[3] = {0., 0., 0.};
    double weight[3] = {0., 0., 0.};
    double all_success = 0., all_expected = 0., count = 0.;
};

static double sigmoid(double x) { return 1. / (1. + std::exp(-x)); }
static double odds_gap(double successes, double expected, double n) {
    return std::log((successes + 1.) / (n - successes + 1.))
         - std::log((expected + 1.) / (n - expected + 1.));
}

static void emit(const State& state, float* out) {
    for (int j = 0; j < 3; ++j) {
        out[j] = static_cast<float>(state.theta[j]);
        out[3+j] = static_cast<float>(odds_gap(state.success[j], state.expected[j], state.weight[j]));
    }
    out[6] = static_cast<float>(odds_gap(state.all_success, state.all_expected, state.count));
}

static void observe(State& state, double correct, double difficulty) {
    const double eta[3] = {.05, .2, .8};
    const double decay[3] = {.5, .9, .98};
    const double prior = sigmoid(-difficulty);
    for (int j = 0; j < 3; ++j) {
        state.theta[j] += eta[j] * (correct - sigmoid(state.theta[j] - difficulty));
        state.success[j] = decay[j] * state.success[j] + correct;
        state.expected[j] = decay[j] * state.expected[j] + prior;
        state.weight[j] = decay[j] * state.weight[j] + 1.;
    }
    state.all_success += correct;
    state.all_expected += prior;
    state.count += 1.;
}

// All output rows precede observation of that row's answer. The inputs include
// fixed training-only item difficulty, never target detector scores.
extern "C" int mica_states(int64_t students, const int64_t* offsets,
        const int32_t* skills, const int32_t* items, const float* correct,
        const float* difficulty, int32_t skill_count, int32_t item_count,
        float* output) {
    std::vector<State> skill_state(skill_count), item_state(item_count);
    std::vector<int64_t> skill_stamp(skill_count, -1), item_stamp(item_count, -1);
    constexpr int width = 28;
    for (int64_t u = 0; u < students; ++u) {
        State global;
        for (int64_t t = offsets[u]; t < offsets[u+1]; ++t) {
            const int32_t k = skills[t], q = items[t];
            if (k < 0 || k >= skill_count || q < 0 || q >= item_count) return 1;
            float* row = output + t * width;
            emit(global, row);
            if (k > 0) {
                if (skill_stamp[k] != u) { skill_state[k] = State(); skill_stamp[k] = u; }
                emit(skill_state[k], row+7);
            }
            if (q > 0) {
                if (item_stamp[q] != u) { item_state[q] = State(); item_stamp[q] = u; }
                emit(item_state[q], row+14);
            }
            row[21] = difficulty[t];
            row[22] = static_cast<float>(global.theta[1] * std::tanh(difficulty[t]));
            row[23] = static_cast<float>(k > 0 ? global.theta[1] * skill_state[k].theta[1] : 0.);
            if (t > offsets[u]) {
                row[24] = k > 0 && k == skills[t-1];
                row[25] = q > 0 && q == items[t-1];
                row[26] = row[24] * (correct[t-1] - .5f);
                row[27] = difficulty[t] - difficulty[t-1];
            }
            observe(global, correct[t], difficulty[t]);
            if (k > 0) observe(skill_state[k], correct[t], difficulty[t]);
            if (q > 0) observe(item_state[q], correct[t], difficulty[t]);
        }
    }
    return 0;
}
