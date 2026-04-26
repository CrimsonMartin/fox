// Exception-safe wrappers for llama.cpp calls that can throw.
// llama.cpp's grammar code throws std::runtime_error in several paths
// (init parsing, accept after empty stack). Letting those cross the
// Rust FFI boundary aborts the process.

#include "llama.h"
#include <cstdio>
#include <exception>

extern "C" {

struct llama_sampler * fox_sampler_init_grammar_safe(
        const struct llama_vocab * vocab,
        const char * grammar_str,
        const char * grammar_root) {
    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (const std::exception & e) {
        fprintf(stderr, "fox: grammar init failed: %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "fox: grammar init failed: unknown exception\n");
        return nullptr;
    }
}

llama_token fox_sampler_sample_safe(
        struct llama_sampler * smpl,
        struct llama_context * ctx,
        int32_t idx) {
    try {
        return llama_sampler_sample(smpl, ctx, idx);
    } catch (const std::exception & e) {
        fprintf(stderr, "fox: sampler_sample failed: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "fox: sampler_sample failed: unknown exception\n");
        return -1;
    }
}

int fox_sampler_accept_safe(
        struct llama_sampler * smpl,
        llama_token token) {
    try {
        llama_sampler_accept(smpl, token);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "fox: sampler_accept failed: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "fox: sampler_accept failed: unknown exception\n");
        return -1;
    }
}

}
