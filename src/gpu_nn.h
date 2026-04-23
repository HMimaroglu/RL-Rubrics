#ifndef GPU_NN_H
#define GPU_NN_H

/* GPU-resident MLP forward pass (compute-shader implementation).
 *
 * One workgroup invocation processes one batch element through the full
 * trunk + policy head + value head in a single shader. Up to max_batch
 * elements per dispatch. Internally allocates SSBOs sized for max_batch.
 *
 * Caller must have a GL 4.3+ context current, and must have called
 * gl_load() to resolve the compute / SSBO function pointers. */

typedef struct GpuMlp GpuMlp;

GpuMlp *gpu_mlp_new(int nx, int nh, int ny, int max_batch);
void    gpu_mlp_free(GpuMlp *m);

/* Upload weights and biases from CPU buffers. Call whenever the CPU
 * params change (every SGD step). Wv and bv may be NULL to skip the
 * value head (in that case GpuMlp will write 0 to all out_values). */
void    gpu_mlp_upload_params(GpuMlp *m,
                               const float *W1, const float *b1,
                               const float *W2, const float *b2,
                               const float *Wv, const float *bv);

/* Run forward pass for batch_size inputs.
 *   inputs           [batch_size * nx]   one input per batch element
 *   forbidden_face   [batch_size]        action mask, -1 for no mask
 * Outputs (filled by readback):
 *   out_h            [batch_size * nh]   post-ReLU hidden
 *   out_probs        [batch_size * ny]   softmax over masked logits
 *   out_values       [batch_size]        scalar V(s) per element
 * Any out_* may be NULL to skip readback of that buffer. */
void    gpu_mlp_forward(GpuMlp *m,
                         int batch_size,
                         const float *inputs,
                         const int   *forbidden_face,
                         float       *out_h,
                         float       *out_probs,
                         float       *out_values);

#endif
