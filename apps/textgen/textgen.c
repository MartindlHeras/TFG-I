/*
  The MIT License

  Copyright (c) 2018-2019 Dana-Farber Cancer Institute
                2016-2018 Broad Institute

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef KANN_AUTODIFF_H
#define KANN_AUTODIFF_H

#define KAD_VERSION "r544"

#include <stdio.h>
#include <stdint.h>

#ifdef __STRICT_ANSI__
#define inline
#endif

#define KAD_MAX_DIM 4     /* max dimension */
#define KAD_MAX_OP  64    /* max number of operators */

/* A computational graph is a directed acyclic graph. In the graph, an external
 * node represents a variable, a constant or a feed; an internal node
 * represents an operator; an edge from node v to w indicates v is an operand
 * of w.
 */

#define KAD_VAR        0x1
#define KAD_CONST      0x2
#define KAD_POOL       0x4
#define KAD_SHARE_RNG  0x10 /* with this flag on, different time step shares the same RNG status after unroll */

#define kad_is_back(p)  ((p)->flag & KAD_VAR)
#define kad_is_ext(p)   ((p)->n_child == 0)
#define kad_is_var(p)   (kad_is_ext(p) && kad_is_back(p))
#define kad_is_const(p) (kad_is_ext(p) && ((p)->flag & KAD_CONST))
#define kad_is_feed(p)  (kad_is_ext(p) && !kad_is_back(p) && !((p)->flag & KAD_CONST))
#define kad_is_pivot(p) ((p)->n_child == 1 && ((p)->flag & KAD_POOL))
#define kad_is_switch(p) ((p)->op == 12 && !((p)->flag & KAD_POOL))
#define kad_use_rng(p)  ((p)->op == 15 || (p)->op == 24)

#define kad_eval_enable(p) ((p)->tmp = 1)
#define kad_eval_disable(p) ((p)->tmp = -1)

/* a node in the computational graph */
typedef struct kad_node_t {
	uint8_t     n_d;            /* number of dimensions; no larger than KAD_MAX_DIM */
	uint8_t     flag;           /* type of the node; see KAD_F_* for valid flags */
	uint16_t    op;             /* operator; kad_op_list[op] is the actual function */
	int32_t     n_child;        /* number of operands/child nodes */
	int32_t     tmp;            /* temporary field; MUST BE zero before calling kad_compile() */
	int32_t     ptr_size;       /* size of ptr below */
	int32_t     d[KAD_MAX_DIM]; /* dimensions */
	int32_t     ext_label;      /* labels for external uses (not modified by the kad_* APIs) */
	uint32_t    ext_flag;       /* flags for external uses (not modified by the kad_* APIs) */
	float      *x;              /* value; allocated for internal nodes */
	float      *g;              /* gradient; allocated for internal nodes */
	void       *ptr;            /* for special operators that need additional parameters (e.g. conv2d) */
	void       *gtmp;           /* temporary data generated at the forward pass but used at the backward pass */
	struct kad_node_t **child;  /* operands/child nodes */
	struct kad_node_t  *pre;    /* usually NULL; only used for RNN */
} kad_node_t, *kad_node_p;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compile/linearize a computational graph
 *
 * @param n_node   number of nodes (out)
 * @param n_roots  number of nodes without predecessors
 * @param roots    list of nodes without predecessors
 *
 * @return list of nodes, of size *n_node
 */
kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots);

kad_node_t **kad_compile(int *n_node, int n_roots, ...); /* an alternative API to above */
void kad_delete(int n, kad_node_t **a); /* deallocate a compiled/linearized graph */

/**
 * Compute the value at a node
 * 
 * @param n       number of nodes
 * @param a       list of nodes
 * @param from    compute the value at this node, 0<=from<n
 *
 * @return a pointer to the value (pointing to kad_node_t::x, so don't call
 *         free() on it!)
 */
const float *kad_eval_at(int n, kad_node_t **a, int from);

void kad_eval_marked(int n, kad_node_t **a);
int kad_sync_dim(int n, kad_node_t **v, int batch_size);

/**
 * Compute gradient
 *
 * @param n       number of nodes
 * @param a       list of nodes
 * @param from    the function node; must be a scalar (compute \nabla a[from])
 */
void kad_grad(int n, kad_node_t **a, int from);

/**
 * Unroll a recurrent computation graph
 *
 * @param n_v     number of nodes
 * @param v       list of nodes
 * @param new_n   number of nodes in the unrolled graph (out)
 * @param len     how many times to unroll, one for each pivot
 *
 * @return list of nodes in the unrolled graph
 */
kad_node_t **kad_unroll(int n_v, kad_node_t **v, int *new_n, int *len);
int kad_n_pivots(int n_v, kad_node_t **v);

kad_node_t **kad_clone(int n, kad_node_t **v, int batch_size);

/* define a variable, a constant or a feed (placeholder in TensorFlow) */
kad_node_t *kad_var(float *x, float *g, int n_d, ...); /* a variable; gradients to be computed; not unrolled */
kad_node_t *kad_const(float *x, int n_d, ...);         /* a constant; no gradients computed; not unrolled */
kad_node_t *kad_feed(int n_d, ...);                    /* an input/output; no gradients computed; unrolled */

/* operators taking two operands */
kad_node_t *kad_add(kad_node_t *x, kad_node_t *y); /* f(x,y) = x + y (generalized element-wise addition; f[i*n+j]=x[i*n+j]+y[j], n=kad_len(y), 0<j<n, 0<i<kad_len(x)/n) */
kad_node_t *kad_sub(kad_node_t *x, kad_node_t *y); /* f(x,y) = x - y (generalized element-wise subtraction) */
kad_node_t *kad_mul(kad_node_t *x, kad_node_t *y); /* f(x,y) = x * y (generalized element-wise product) */

kad_node_t *kad_matmul(kad_node_t *x, kad_node_t *y);     /* f(x,y) = x * y   (general matrix product) */
kad_node_t *kad_cmul(kad_node_t *x, kad_node_t *y);       /* f(x,y) = x * y^T (column-wise matrix product; i.e. y is transposed) */

/* loss functions; output scalar */
kad_node_t *kad_mse(kad_node_t *x, kad_node_t *y);        /* mean square error */
kad_node_t *kad_ce_multi(kad_node_t *x, kad_node_t *y);   /* multi-class cross-entropy; x is the preidction and y is the truth */
kad_node_t *kad_ce_bin(kad_node_t *x, kad_node_t *y);     /* binary cross-entropy for (0,1) */
kad_node_t *kad_ce_bin_neg(kad_node_t *x, kad_node_t *y); /* binary cross-entropy for (-1,1) */
kad_node_t *kad_ce_multi_weighted(kad_node_t *pred, kad_node_t *truth, kad_node_t *weight);

#define KAD_PAD_NONE  0      /* use the smallest zero-padding */
#define KAD_PAD_SAME  (-2)   /* output to have the same dimension as input */

kad_node_t *kad_conv2d(kad_node_t *x, kad_node_t *w, int r_stride, int c_stride, int r_pad, int c_pad);             /* 2D convolution with weight matrix flipped */
kad_node_t *kad_max2d(kad_node_t *x, int kernel_h, int kernel_w, int r_stride, int c_stride, int r_pad, int c_pad); /* 2D max pooling */
kad_node_t *kad_conv1d(kad_node_t *x, kad_node_t *w, int stride, int pad);  /* 1D convolution with weight flipped */
kad_node_t *kad_max1d(kad_node_t *x, int kernel_size, int stride, int pad); /* 1D max pooling */
kad_node_t *kad_avg1d(kad_node_t *x, int kernel_size, int stride, int pad); /* 1D average pooling */

kad_node_t *kad_dropout(kad_node_t *x, kad_node_t *r);                      /* dropout at rate r */
kad_node_t *kad_sample_normal(kad_node_t *x);                               /* f(x) = x * r, where r is drawn from a standard normal distribution */

/* operators taking one operand */
kad_node_t *kad_square(kad_node_t *x); /* f(x) = x^2                         (element-wise square) */
kad_node_t *kad_sigm(kad_node_t *x);   /* f(x) = 1/(1+exp(-x))               (element-wise sigmoid) */
kad_node_t *kad_tanh(kad_node_t *x);   /* f(x) = (1-exp(-2x)) / (1+exp(-2x)) (element-wise tanh) */
kad_node_t *kad_relu(kad_node_t *x);   /* f(x) = max{0,x}                    (element-wise rectifier, aka ReLU) */
kad_node_t *kad_softmax(kad_node_t *x);/* f_i(x_1,...,x_n) = exp(x_i) / \sum_j exp(x_j) (softmax: tf.nn.softmax(x,dim=-1)) */
kad_node_t *kad_1minus(kad_node_t *x); /* f(x) = 1 - x */
kad_node_t *kad_exp(kad_node_t *x);    /* f(x) = exp(x) */
kad_node_t *kad_log(kad_node_t *x);    /* f(x) = log(x) */
kad_node_t *kad_sin(kad_node_t *x);    /* f(x) = sin(x) */

kad_node_t *kad_stdnorm(kad_node_t *x); /* layer normalization; applied to the last dimension */

/* operators taking an indefinite number of operands (e.g. pooling) */
kad_node_t *kad_avg(int n, kad_node_t **x);   /* f(x_1,...,x_n) = \sum_i x_i/n      (mean pooling) */
kad_node_t *kad_max(int n, kad_node_t **x);   /* f(x_1,...,x_n) = max{x_1,...,x_n}  (max pooling) */
kad_node_t *kad_stack(int n, kad_node_t **x); /* f(x_1,...,x_n) = [x_1,...,x_n]     (stack pooling) */
kad_node_t *kad_select(int n, kad_node_t **x, int which); /* f(x_1,...,x_n;i) = x_i (select pooling; -1 for the last) */

/* dimension reduction */
kad_node_t *kad_reduce_sum(kad_node_t *x, int axis);  /* tf.reduce_sum(x, axis) */
kad_node_t *kad_reduce_mean(kad_node_t *x, int axis); /* tf.reduce_mean(x, axis) */

/* special operators */
kad_node_t *kad_slice(kad_node_t *x, int axis, int start, int end); /* take a slice on the axis-th dimension */
kad_node_t *kad_concat(int axis, int n, ...);                       /* concatenate on the axis-th dimension */
kad_node_t *kad_concat_array(int axis, int n, kad_node_t **p);      /* the array version of concat */
kad_node_t *kad_reshape(kad_node_t *x, int n_d, int *d);            /* reshape; similar behavior to TensorFlow's reshape() */
kad_node_t *kad_reverse(kad_node_t *x, int axis);
kad_node_t *kad_switch(int n, kad_node_t **p);                      /* manually (as a hyperparameter) choose one input, default to 0 */

/* miscellaneous operations on a compiled graph */
int kad_size_var(int n, kad_node_t *const* v);   /* total size of all variables */
int kad_size_const(int n, kad_node_t *const* v); /* total size of all constants */

/* graph I/O */
int kad_save(FILE *fp, int n_node, kad_node_t **node);
kad_node_t **kad_load(FILE *fp, int *_n_node);

/* random number generator */
void *kad_rng(void);
void kad_srand(void *d, uint64_t seed);
uint64_t kad_rand(void *d);
double kad_drand(void *d);
double kad_drand_normal(void *d);
void kad_saxpy(int n, float a, const float *x, float *y);

/* debugging routines */
void kad_trap_fe(void); /* abort on divide-by-zero and NaN */
void kad_print_graph(FILE *fp, int n, kad_node_t **v);
void kad_check_grad(int n, kad_node_t **a, int from);

#ifdef __cplusplus
}
#endif

#define KAD_ALLOC      1
#define KAD_FORWARD    2
#define KAD_BACKWARD   3
#define KAD_SYNC_DIM   4

typedef int (*kad_op_f)(kad_node_t*, int);
extern kad_op_f kad_op_list[KAD_MAX_OP];
extern char *kad_op_name[KAD_MAX_OP];

static inline int kad_len(const kad_node_t *p) /* calculate the size of p->x */
{
	int n = 1, i;
	for (i = 0; i < p->n_d; ++i) n *= p->d[i];
	return n;
}

#endif

#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>
#include <math.h>

typedef struct {
	uint64_t s[2];
	double n_gset;
	int n_iset;
	volatile int lock;
} kad_rng_t;

/**********************
 * Graph construction *
 **********************/

static inline kad_node_t *kad_new_core(int n_d, int op, int n_child)
{
	kad_node_t *s;
	if (n_d >= KAD_MAX_DIM) return 0;
	s = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	s->n_d = n_d, s->op = op, s->n_child = n_child;
	if (s->n_child) s->child = (kad_node_t**)calloc(s->n_child, sizeof(kad_node_t*));
	return s;
}

static inline kad_node_t *kad_vleaf(uint8_t flag, float *x, float *g, int n_d, va_list ap)
{
	int i;
	kad_node_t *p;
	if (n_d > KAD_MAX_DIM) return 0;
	p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	p->n_d = n_d;
	for (i = 0; i < n_d; ++i)
		p->d[i] = va_arg(ap, int32_t);
	p->x = x, p->g = g, p->flag = flag;
	return p;
}

kad_node_t *kad_const(float *x, int n_d, ...)
{
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d); p = kad_vleaf(KAD_CONST, x, 0, n_d, ap); va_end(ap);
	return p;
}

kad_node_t *kad_feed(int n_d, ...)
{
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d); p = kad_vleaf(0, 0, 0, n_d, ap); va_end(ap);
	return p;
}

kad_node_t *kad_var(float *x, float *g, int n_d, ...)
{
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d); p = kad_vleaf(KAD_VAR, x, g, n_d, ap); va_end(ap);
	return p;
}

static inline kad_node_t *kad_finalize_node(kad_node_t *s) /* a helper function */
{
	int i;
	if (kad_op_list[s->op](s, KAD_SYNC_DIM) < 0) { /* check dimension */
		if (s->ptr) free(s->ptr);
		free(s->child); free(s);
		return 0;
	}
	for (i = 0; i < s->n_child; ++i)
		if (kad_is_back(s->child[i]))
			break;
	if (i < s->n_child) s->flag |= KAD_VAR;
	return s;
}

/********** Simple arithmetic **********/

static inline kad_node_t *kad_op2_core(int op, kad_node_t *x, kad_node_t *y)
{
	kad_node_t *s;
	s = kad_new_core(0, op, 2);
	s->child[0] = x, s->child[1] = y;
	return kad_finalize_node(s);
}

static inline kad_node_t *kad_op1_core(int op, kad_node_t *x)
{
	kad_node_t *s;
	s = kad_new_core(0, op, 1);
	s->child[0] = x;
	return kad_finalize_node(s);
}

#define KAD_FUNC_OP2(fname, op) kad_node_t *fname(kad_node_t *x, kad_node_t *y) { return kad_op2_core((op), x, y); }

KAD_FUNC_OP2(kad_add, 1)
KAD_FUNC_OP2(kad_sub, 23)
KAD_FUNC_OP2(kad_mul, 2)
KAD_FUNC_OP2(kad_cmul, 3)
KAD_FUNC_OP2(kad_matmul, 9)
KAD_FUNC_OP2(kad_ce_multi, 13)
KAD_FUNC_OP2(kad_ce_bin, 22)
KAD_FUNC_OP2(kad_ce_bin_neg, 4)
KAD_FUNC_OP2(kad_mse, 29)

#define KAD_FUNC_OP1(fname, op) kad_node_t *fname(kad_node_t *x) { return kad_op1_core((op), x); }

KAD_FUNC_OP1(kad_log, 27)
KAD_FUNC_OP1(kad_exp, 33)
KAD_FUNC_OP1(kad_sin, 34)
KAD_FUNC_OP1(kad_square, 5)
KAD_FUNC_OP1(kad_sigm, 6)
KAD_FUNC_OP1(kad_tanh, 7)
KAD_FUNC_OP1(kad_relu, 8)
KAD_FUNC_OP1(kad_1minus, 11)
KAD_FUNC_OP1(kad_softmax, 14)
KAD_FUNC_OP1(kad_stdnorm, 32)

kad_node_t *kad_ce_multi_weighted(kad_node_t *pred, kad_node_t *truth, kad_node_t *weight)
{
	kad_node_t *s;
	s = kad_new_core(0, 13, 3);
	s->child[0] = pred, s->child[1] = truth, s->child[2] = weight;
	return kad_finalize_node(s);
}

/********** Convolution **********/

/* compute output dimension and padding sizes on both sides */
static inline int conv_find_par(int in_size, int kernel_size, int stride, int pad0, int *new_pad0, int *new_pad1)
{
	int out_size, pad_both;
	/* key equation: out_size = (in_size - kernel_size + pad_both) / stride + 1 */
	if (pad0 == KAD_PAD_SAME && stride == 1) out_size = in_size;
	else out_size = (in_size - kernel_size + (pad0 > 0? pad0 : 0) + stride - 1) / stride + 1;
	pad_both = (out_size - 1) * stride + kernel_size - in_size;
	*new_pad0 = pad_both / 2;
	*new_pad1 = pad_both - *new_pad0;
	return out_size;
}

typedef struct {
	int kernel_size, stride, pad[2];
} conv_conf_t;

static inline conv_conf_t *conv2d_gen_aux(int in_row, int in_col, int kernel_r, int kernel_c, int stride_r, int stride_c, int top_pad, int left_pad)
{
	conv_conf_t *cnn;
	cnn = (conv_conf_t*)calloc(2, sizeof(conv_conf_t));
	cnn[0].kernel_size = kernel_r, cnn[0].stride = stride_r;
	cnn[1].kernel_size = kernel_c, cnn[1].stride = stride_c;
	conv_find_par(in_row, kernel_r, stride_r, top_pad,  &cnn[0].pad[0], &cnn[0].pad[1]);
	conv_find_par(in_col, kernel_c, stride_c, left_pad, &cnn[1].pad[0], &cnn[1].pad[1]);
	return cnn;
}

kad_node_t *kad_conv2d(kad_node_t *x, kad_node_t *w, int stride_r, int stride_c, int top_pad, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 4 || w->n_d != 4) return 0;
	s = kad_new_core(0, 16, 2);
	s->child[0] = x, s->child[1] = w;
	s->ptr = conv2d_gen_aux(x->d[2], x->d[3], w->d[2], w->d[3], stride_r, stride_c, top_pad, left_pad);
	s->ptr_size = sizeof(conv_conf_t) * 2;
	return kad_finalize_node(s);
}

kad_node_t *kad_max2d(kad_node_t *x, int kernel_r, int kernel_c, int stride_r, int stride_c, int top_pad, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 4) return 0;
	s = kad_new_core(0, 17, 1);
	s->child[0] = x;
	s->ptr = conv2d_gen_aux(x->d[2], x->d[3], kernel_r, kernel_c, stride_r, stride_c, top_pad, left_pad);
	s->ptr_size = sizeof(conv_conf_t) * 2;
	return kad_finalize_node(s);
}

static inline conv_conf_t *conv1d_gen_aux(int in_col, int kernel_c, int stride_c, int left_pad)
{
	conv_conf_t *cnn;
	cnn = (conv_conf_t*)calloc(1, sizeof(conv_conf_t));
	cnn->kernel_size = kernel_c, cnn->stride = stride_c;
	conv_find_par(in_col, kernel_c, stride_c, left_pad, &cnn->pad[0], &cnn->pad[1]);
	return cnn;
}

kad_node_t *kad_conv1d(kad_node_t *x, kad_node_t *w, int stride, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 3 || w->n_d != 3) return 0;
	s = kad_new_core(0, 18, 2);
	s->child[0] = x, s->child[1] = w;
	s->ptr = conv1d_gen_aux(x->d[2], w->d[2], stride, left_pad);
	s->ptr_size = sizeof(conv_conf_t);
	return kad_finalize_node(s);
}

kad_node_t *kad_max1d(kad_node_t *x, int kernel_size, int stride, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 3) return 0;
	s = kad_new_core(0, 19, 1);
	s->child[0] = x;
	s->ptr = conv1d_gen_aux(x->d[2], kernel_size, stride, left_pad);
	s->ptr_size = sizeof(conv_conf_t);
	return kad_finalize_node(s);
}

kad_node_t *kad_avg1d(kad_node_t *x, int kernel_size, int stride, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 3) return 0;
	s = kad_new_core(0, 28, 1);
	s->child[0] = x;
	s->ptr = conv1d_gen_aux(x->d[2], kernel_size, stride, left_pad);
	s->ptr_size = sizeof(conv_conf_t);
	return kad_finalize_node(s);
}

/********** Multi-node pooling **********/

static kad_node_t *kad_pooling_general(int op, int n, kad_node_t **x)
{
	int i;
	kad_node_t *s;
	s = kad_new_core(0, op, n);
	s->flag |= KAD_POOL;
	for (i = 0; i < n; ++i)
		s->child[i] = x[i];
	return kad_finalize_node(s);
}

kad_node_t *kad_avg(int n, kad_node_t **x)   { return kad_pooling_general(10, n, x); }
kad_node_t *kad_max(int n, kad_node_t **x)   { return kad_pooling_general(21, n, x); }
kad_node_t *kad_stack(int n, kad_node_t **x) { return kad_pooling_general(35, n, x); }

kad_node_t *kad_select(int n, kad_node_t **x, int which)
{
	kad_node_t *s;
	int32_t i, *aux;
	aux = (int32_t*)calloc(1, 4);
	*aux = which;
	s = kad_new_core(0, 12, n);
	for (i = 0; i < n; ++i) s->child[i] = x[i];
	s->flag |= KAD_POOL, s->ptr = aux, s->ptr_size = 4;
	return kad_finalize_node(s);
}

/********** Dimension reduction **********/

static kad_node_t *kad_reduce_general(int op, kad_node_t *x, int axis)
{
	kad_node_t *s;
	int32_t *aux;
	aux = (int32_t*)malloc(4);
	aux[0] = axis;
	s = kad_new_core(0, op, 1);
	s->child[0] = x;
	s->ptr = aux, s->ptr_size = 4;
	return kad_finalize_node(s);
}

kad_node_t *kad_reduce_sum(kad_node_t *x, int axis)  { return kad_reduce_general(25, x, axis); }
kad_node_t *kad_reduce_mean(kad_node_t *x, int axis) { return kad_reduce_general(26, x, axis); }

/********** Sampling related **********/

kad_node_t *kad_dropout(kad_node_t *x, kad_node_t *y)
{
	kad_node_t *z;
	z = kad_op2_core(15, x, y);
	z->ptr = kad_rng(), z->ptr_size = sizeof(kad_rng_t);
	return z;
}

kad_node_t *kad_sample_normal(kad_node_t *x)
{
	kad_node_t *z;
	z = kad_op1_core(24, x);
	z->ptr = kad_rng(), z->ptr_size = sizeof(kad_rng_t);
	return z;
}

/********** Miscellaneous **********/

kad_node_t *kad_slice(kad_node_t *x, int axis, int start, int end)
{
	kad_node_t *s;
	int32_t *aux;
	if (end < start || start < 0) return 0;
	aux = (int32_t*)malloc(3 * 4);
	aux[0] = axis, aux[1] = start, aux[2] = end;
	s = kad_new_core(0, 20, 1);
	s->child[0] = x;
	s->ptr = aux, s->ptr_size = 3 * 4;
	return kad_finalize_node(s);
}

kad_node_t *kad_concat_array(int axis, int n, kad_node_t **p)
{
	kad_node_t *s;
	int32_t i, *aux;
	aux = (int32_t*)malloc(4);
	aux[0] = axis;
	s = kad_new_core(0, 31, n);
	for (i = 0; i < n; ++i)
		s->child[i] = p[i];
	s->ptr = aux, s->ptr_size = 4;
	return kad_finalize_node(s);
}

kad_node_t *kad_concat(int axis, int n, ...)
{
	int i;
	kad_node_t **p, *s;
	va_list ap;
	p = (kad_node_t**)malloc(n * sizeof(kad_node_t*));
	va_start(ap, n);
	for (i = 0; i < n; ++i) p[i] = va_arg(ap, kad_node_p);
	va_end(ap);
	s = kad_concat_array(axis, n, p);
	free(p);
	return s;
}

kad_node_t *kad_reshape(kad_node_t *x, int n_d, int *d)
{
	kad_node_t *s;
	int32_t i, *aux = 0;
	if (n_d > 0) {
		aux = (int32_t*)malloc(n_d * 4);
		for (i = 0; i < n_d; ++i) aux[i] = d? d[i] : -1;
	}
	s = kad_new_core(0, 30, 1);
	s->child[0] = x, s->ptr = aux, s->ptr_size = n_d * 4;
	return kad_finalize_node(s);
}

kad_node_t *kad_reverse(kad_node_t *x, int axis)
{
	kad_node_t *s;
	int32_t *aux;
	aux = (int32_t*)malloc(4);
	*aux = axis;
	s = kad_new_core(0, 36, 1);
	s->child[0] = x, s->ptr = aux, s->ptr_size = 4;
	return kad_finalize_node(s);
}

kad_node_t *kad_switch(int n, kad_node_t **p)
{
	kad_node_t *s;
	int32_t i, *aux;
	aux = (int32_t*)calloc(1, 4);
	s = kad_new_core(0, 12, n);
	for (i = 0; i < n; ++i)
		s->child[i] = p[i];
	s->ptr = aux, s->ptr_size = 4;
	return kad_finalize_node(s);
}

/***********************
 * Graph linearization *
 ***********************/

static void kad_mark_back(int n, kad_node_t **v)
{
	int i, j;
	for (i = 0; i < n; ++i) {
		if (v[i]->n_child == 0) continue;
		for (j = 0; j < v[i]->n_child; ++j)
			if (kad_is_back(v[i]->child[j]))
				break;
		if (j < v[i]->n_child) v[i]->flag |= KAD_VAR;
		else v[i]->flag &= ~KAD_VAR;
	}
}

static void kad_allocate_internal(int n, kad_node_t **v)
{
	int i;
	kad_mark_back(n, v);
	for (i = 0; i < n; ++i) {
		kad_node_t *p = v[i];
		if (p->n_child == 0) continue;
		p->x = (float*)realloc(p->x, kad_len(p) * sizeof(float));
		if (kad_is_back(p)) {
			p->g = (float*)realloc(p->g, kad_len(p) * sizeof(float));
			kad_op_list[p->op](p, KAD_ALLOC);
		}
	}
}

int kad_sync_dim(int n, kad_node_t **v, int batch_size)
{
	int i, req_alloc = 0, req_sync = 0, old_size = 0;
	for (i = 0; i < n; ++i) {
		if (kad_is_feed(v[i])) {
			old_size = v[i]->d[0]; /* TODO: check if all feeds have the same batch size */
			if (batch_size > 0 && v[i]->d[0] != batch_size)
				v[i]->d[0] = batch_size, req_sync = 1;
		} else if (v[i]->n_child > 0 && req_sync)
			kad_op_list[v[i]->op](v[i], KAD_SYNC_DIM);
	}
	if (old_size < batch_size) req_alloc = 1;
	for (i = 0; i < n; ++i)
		if (v[i]->n_child > 0 && v[i]->x == 0) req_alloc = 1;
	if (req_alloc) kad_allocate_internal(n, v);
	return batch_size > 0? batch_size : old_size;
}

#define kvec_t(type) struct { size_t n, m; type *a; }

#define kv_pop(v) ((v).a[--(v).n])

#define kv_push(type, v, x) do { \
		if ((v).n == (v).m) { \
			(v).m = (v).m? (v).m<<1 : 2; \
			(v).a = (type*)realloc((v).a, sizeof(type) * (v).m); \
		} \
		(v).a[(v).n++] = (x); \
	} while (0)

/* IMPORTANT: kad_node_t::tmp MUST BE set to zero before calling this function */
kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots)
{
	int i;
	kvec_t(kad_node_p) stack = {0,0,0}, a = {0,0,0};

	/* generate kad_node_t::tmp, the count of the parent nodes; shifted by 1; lowest bit to detect fake roots */
	for (i = 0; i < n_roots; ++i) {
		roots[i]->tmp = 1; /* mark the root */
		kv_push(kad_node_p, stack, roots[i]);
	}
	while (stack.n) {
		kad_node_t *p = kv_pop(stack);
		for (i = 0; i < p->n_child; ++i) {
			kad_node_t *q = p->child[i];
			if (q->tmp == 0) kv_push(kad_node_p, stack, q);
			q->tmp += 1<<1;
		}
	}

	/* topological sorting (Kahn's algorithm) */
	for (i = 0; i < n_roots; ++i)
		if (roots[i]->tmp>>1 == 0) /* if roots[i]->tmp>>1 != 0, it is not a real root */
			kv_push(kad_node_p, stack, roots[i]);
	while (stack.n) {
		kad_node_t *p = kv_pop(stack);
		kv_push(kad_node_p, a, p);
		for (i = 0; i < p->n_child; ++i) {
			p->child[i]->tmp -= 1<<1;
			if (p->child[i]->tmp>>1 == 0)
				kv_push(kad_node_p, stack, p->child[i]);
		}
	}
	free(stack.a);
	for (i = 0; i < (int)a.n; ++i) { /* check cycles; no cycles if constructed with kad_add() etc */
		assert(a.a[i]->tmp>>1 == 0);
		a.a[i]->tmp = 0;
	}

	/* reverse */
	for (i = 0; i < (int)a.n>>1; ++i) { /* reverse a.a[] */
		kad_node_p t;
		t = a.a[i], a.a[i] = a.a[a.n-1-i], a.a[a.n-1-i] = t;
	}
	kad_allocate_internal(a.n, a.a);

	*n_node = a.n;
	return a.a;
}

kad_node_t **kad_compile(int *n_node, int n_roots, ...)
{
	int i;
	kad_node_t **roots, **ret;
	va_list ap;

	roots = (kad_node_t**)malloc(n_roots * sizeof(kad_node_t*));
	va_start(ap, n_roots);
	for (i = 0; i < n_roots; ++i) roots[i] = va_arg(ap, kad_node_p);
	va_end(ap);
	ret = kad_compile_array(n_node, n_roots, roots);
	free(roots);
	return ret;
}

/************************************
 * Miscellaneous on compiled graphs *
 ************************************/

void kad_delete(int n, kad_node_t **a)
{
	int i;
	for (i = 0; i < n; ++i) {
		kad_node_t *p = a[i];
		if (p->n_child) {
			free(p->x); free(p->g);
		}
		free(p->child); free(p->ptr); free(p->gtmp); free(p);
	}
	free(a);
}

int kad_size_var(int n, kad_node_t *const* v)
{
	int c, i;
	for (i = c = 0; i < n; ++i)
		if (kad_is_var(v[i]))
			c += kad_len(v[i]);
	return c;
}

int kad_size_const(int n, kad_node_t *const* v)
{
	int c, i;
	for (i = c = 0; i < n; ++i)
		if (kad_is_const(v[i]))
			c += kad_len(v[i]);
	return c;
}

/**********************************
 * Computate values and gradients *
 **********************************/

static void kad_propagate_marks(int n, kad_node_t **a)
{
	int i, j;
	for (i = n - 1; i >= 0; --i) {
		kad_node_t *p = a[i];
		if (p->tmp > 0) {
			if (kad_is_switch(p)) {
				int32_t *aux = (int32_t*)p->ptr;
				if (p->child[*aux]->tmp == 0)
					p->child[*aux]->tmp = 1;
			} else {
				for (j = 0; j < p->n_child; ++j)
					if (p->child[j]->tmp == 0)
						p->child[j]->tmp = 1;
			}
		}
	}
}

void kad_eval_marked(int n, kad_node_t **a)
{
	int i;
	kad_propagate_marks(n, a);
	for (i = 0; i < n; ++i)
		if (a[i]->n_child && a[i]->tmp > 0)
			kad_op_list[a[i]->op](a[i], KAD_FORWARD);
	for (i = 0; i < n; ++i) a[i]->tmp = 0;
}

const float *kad_eval_at(int n, kad_node_t **a, int from)
{
	int i;
	if (from < 0 || from >= n) from = n - 1;
	for (i = 0; i < n; ++i) a[i]->tmp = (i == from);
	kad_eval_marked(n, a);
	return a[from]->x;
}

void kad_grad(int n, kad_node_t **a, int from)
{
	int i;
	if (from < 0 || from >= n) from = n - 1;
	assert(a[from]->n_d == 0);
	for (i = 0; i < n; ++i) a[i]->tmp = (i == from);
	kad_propagate_marks(n, a);
	for (i = 0; i <= from; ++i) /* set all grandients to zero */
		if (a[i]->g && a[i]->tmp > 0)
			memset(a[i]->g, 0, kad_len(a[i]) * sizeof(float));
	for (i = from, a[i]->g[0] = 1.0f; i >= 0; --i) /* backprop */
		if (a[i]->n_child && a[i]->tmp > 0)
			kad_op_list[a[i]->op](a[i], KAD_BACKWARD);
	for (i = 0; i <= from; ++i) a[i]->tmp = 0;
}

/***********************
 * Load and save graph *
 ***********************/

static void kad_save1(FILE *fp, const kad_node_t *p)
{
	fwrite(&p->ext_label, 4, 1, fp);
	fwrite(&p->ext_flag, 4, 1, fp);
	fwrite(&p->flag, 1, 1, fp);
	fwrite(&p->n_child, 4, 1, fp);
	if (p->n_child) {
		int32_t j, pre = p->pre? p->pre->tmp : -1;
		fwrite(&p->op, 2, 1, fp);
		for (j = 0; j < p->n_child; ++j)
			fwrite(&p->child[j]->tmp, 4, 1, fp);
		fwrite(&pre, 4, 1, fp);
		fwrite(&p->ptr_size, 4, 1, fp);
		if (p->ptr_size > 0 && p->ptr)
			fwrite(p->ptr, p->ptr_size, 1, fp);
	} else {
		fwrite(&p->n_d, 1, 1, fp);
		if (p->n_d) fwrite(p->d, 4, p->n_d, fp);
	}
}

static kad_node_t *kad_load1(FILE *fp, kad_node_t **node)
{
	kad_node_t *p;
	p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	fread(&p->ext_label, 4, 1, fp);
	fread(&p->ext_flag, 4, 1, fp);
	fread(&p->flag, 1, 1, fp);
	fread(&p->n_child, 4, 1, fp);
	if (p->n_child) {
		int32_t j, k;
		p->child = (kad_node_t**)calloc(p->n_child, sizeof(kad_node_t*));
		fread(&p->op, 2, 1, fp);
		for (j = 0; j < p->n_child; ++j) {
			fread(&k, 4, 1, fp);
			p->child[j] = node? node[k] : 0;
		}
		fread(&k, 4, 1, fp);
		if (k >= 0) p->pre = node[k];
		fread(&p->ptr_size, 4, 1, fp);
		if (p->ptr_size > 0) {
			p->ptr = malloc(p->ptr_size);
			fread(p->ptr, p->ptr_size, 1, fp);
		}
	} else {
		fread(&p->n_d, 1, 1, fp);
		if (p->n_d) fread(p->d, 4, p->n_d, fp);
	}
	return p;
}

int kad_save(FILE *fp, int n_node, kad_node_t **node)
{
	int32_t i, k = n_node;
	fwrite(&k, 4, 1, fp);
	for (i = 0; i < n_node; ++i) node[i]->tmp = i;
	for (i = 0; i < n_node; ++i) kad_save1(fp, node[i]);
	for (i = 0; i < n_node; ++i) node[i]->tmp = 0;
	return 0;
}

kad_node_t **kad_load(FILE *fp, int *_n_node)
{
	int32_t i, n_node;
	kad_node_t **node;
	fread(&n_node, 4, 1, fp);
	node = (kad_node_t**)malloc(n_node * sizeof(kad_node_t*));
	for (i = 0; i < n_node; ++i) {
		kad_node_t *p;
		p = node[i] = kad_load1(fp, node);
		if (p->n_child) {
			kad_op_list[p->op](p, KAD_ALLOC);
			kad_op_list[p->op](p, KAD_SYNC_DIM);
		}
	}
	*_n_node = n_node;
	kad_mark_back(n_node, node);
	return node;
}

/***************
 * Graph clone *
 ***************/

static inline kad_node_t *kad_dup1(const kad_node_t *p)
{
	kad_node_t *q;
	q = (kad_node_t*)malloc(sizeof(kad_node_t));
	memcpy(q, p, sizeof(kad_node_t));
	q->pre = 0, q->tmp = 0, q->gtmp = 0;
	if (p->ptr && p->ptr_size > 0) {
		if (kad_use_rng(p) && !(p->flag & KAD_SHARE_RNG) && p->ptr_size == sizeof(kad_rng_t)) {
			q->ptr = kad_rng(); /* each time step uses a different RNG */
		} else {
			q->ptr = malloc(p->ptr_size);
			memcpy(q->ptr, p->ptr, p->ptr_size);
		}
	}
	if (q->n_child) {
		q->x = q->g = 0;
		q->child = (kad_node_t**)calloc(q->n_child, sizeof(kad_node_t*));
	}
	return q;
}

kad_node_t **kad_clone(int n, kad_node_t **v, int batch_size)
{
	int i, j;
	kad_node_t **u;
	u = (kad_node_t**)calloc(n, sizeof(kad_node_t*));
	for (i = 0; i < n; ++i) v[i]->tmp = i;
	for (i = 0; i < n; ++i) {
		kad_node_t *p = v[i], *q;
		q = u[i] = kad_dup1(p);
		if (p->pre) q->pre = u[p->pre->tmp];
		if (p->n_child) {
			for (j = 0; j < p->n_child; ++j)
				q->child[j] = u[p->child[j]->tmp];
		} else if (!kad_is_feed(p)) {
			q->x = (float*)malloc(kad_len(p) * sizeof(float));
			memcpy(q->x, p->x, kad_len(p) * sizeof(float));
			q->g = 0;
		}
	}
	for (i = 0; i < n; ++i) v[i]->tmp = 0;
	kad_sync_dim(n, u, batch_size); /* this will allocate x[] and g[] at internal nodes */
	return u;
}

/**************
 * Unroll RNN *
 **************/

typedef struct {
	int32_t n, m;
	kad_node_t **v;
} nodes_t;

static inline void push_nodes(nodes_t *w, kad_node_t *p)
{
	if (w->n == w->m) {
		w->m = w->m? w->m<<1 : 16;
		w->v = (kad_node_t**)realloc(w->v, w->m * sizeof(kad_node_t*));
	}
	w->v[w->n++] = p;
}

static void kad_unroll_helper(int n_v, kad_node_t **v, int i_pivot, kad_node_t **t, int len, nodes_t *w)
{
	int i, j, l;
	uint8_t *flag;
	kad_node_t **aux;

	assert(kad_is_pivot(v[i_pivot]) && t[i_pivot] == 0);
	t[i_pivot] = kad_dup1(v[i_pivot]);
	t[i_pivot]->n_child = len;
	t[i_pivot]->child = (kad_node_t**)realloc(t[i_pivot]->child, len * sizeof(kad_node_t*));

	flag = (uint8_t*)calloc(n_v, 1);
	for (i = i_pivot, flag[i] = 16; i >= 0; --i) {
		if (i < i_pivot && kad_is_pivot(v[i])) continue; /* don't trespass other pivots */
		if (flag[i]&16) /* flag 16: nodes to unroll */
			for (j = 0; j < v[i]->n_child; ++j)
				flag[v[i]->child[j]->tmp] = 16;
	}
	for (i = 0; i < i_pivot; ++i) {
		if (!(flag[i]&16)) continue;
		if (kad_is_var(v[i]) || kad_is_const(v[i]) || kad_is_pivot(v[i])) flag[i] |= 1; /* external nodes that should not be duplicated */
		if (v[i]->pre) flag[v[i]->pre->tmp] |= 2;
	}
	flag[v[i_pivot]->child[0]->tmp] |= 4;
	aux = (kad_node_t**)calloc(n_v, sizeof(kad_node_t*));
	for (l = 0; l < len; ++l) {
		for (i = 0; i < i_pivot; ++i) {
			if (!(flag[i]&16) || ((flag[i]&3) && t[i])) continue;
			t[i] = kad_dup1(v[i]);
			if (v[i]->n_child)
				for (j = 0; j < v[i]->n_child; ++j)
					t[i]->child[j] = t[v[i]->child[j]->tmp];
			if (flag[i]&4) t[i_pivot]->child[l] = t[i];
			if (l == 0 && (flag[i]&2)) aux[i] = t[i];
			if (v[i]->pre) {
				t[v[i]->pre->tmp] = t[i];
				if (l == len - 1) t[i]->pre = aux[v[i]->pre->tmp]; /* this forms a cycle! */
			}
			push_nodes(w, t[i]);
		}
	}
	push_nodes(w, t[i_pivot]);
	free(aux); free(flag);
}

int kad_n_pivots(int n_v, kad_node_t **v)
{
	int i, n_pivots = 0;
	for (i = 0; i < n_v; ++i)
		if (kad_is_pivot(v[i])) ++n_pivots;
	return n_pivots;
}

kad_node_t **kad_unroll(int n_v, kad_node_t **v, int *new_n, int *len)
{
	int i, j, n_pivots = 0;
	kad_node_t **t;
	nodes_t w = {0,0,0};

	t = (kad_node_t**)calloc(n_v, sizeof(kad_node_t*));
	n_pivots = kad_n_pivots(n_v, v);
	for (i = 0; i < n_v; ++i) v[i]->tmp = i;
	if (n_pivots) {
		int k, *i_pivots;
		i_pivots = (int*)calloc(n_pivots, sizeof(int));
		for (i = k = 0; i < n_v; ++i) /* collect pivots */
			if (kad_is_pivot(v[i])) i_pivots[k++] = i;
		for (i = 0; i < n_pivots; ++i) /* unroll each pivot, from the lowest to the highest */
			kad_unroll_helper(n_v, v, i_pivots[i], t, len[i], &w);
		free(i_pivots);
	}
	for (i = 0; i < n_v; ++i) { /* copy over the rest of nodes */
		if (t[i]) continue;
		t[i] = kad_dup1(v[i]);
		if (v[i]->n_child)
			for (j = 0; j < v[i]->n_child; ++j)
				t[i]->child[j] = t[v[i]->child[j]->tmp];
		push_nodes(&w, t[i]);
	}
	free(t);
	for (i = 0; i < n_v; ++i) v[i]->tmp = 0;
	for (i = 0; i < w.n; ++i) /* stack may change the output dimension */
		if (w.v[i]->n_child > 0)
			kad_op_list[w.v[i]->op](w.v[i], KAD_SYNC_DIM);
	kad_allocate_internal(w.n, w.v);
	*new_n = w.n;
	return w.v;
}

/********************************
 * Vector and matrix operations *
 ********************************/

#ifdef __SSE__
#include <xmmintrin.h>

static inline float kad_sdot(int n, const float *x, const float *y) /* BLAS sdot using SSE */
{
	int i, n8 = n>>3<<3;
	__m128 vs1, vs2;
	float s, t[4];
	vs1 = _mm_setzero_ps();
	vs2 = _mm_setzero_ps();
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2;
		vx1 = _mm_loadu_ps(&x[i]);
		vx2 = _mm_loadu_ps(&x[i+4]);
		vy1 = _mm_loadu_ps(&y[i]);
		vy2 = _mm_loadu_ps(&y[i+4]);
		vs1 = _mm_add_ps(vs1, _mm_mul_ps(vx1, vy1));
		vs2 = _mm_add_ps(vs2, _mm_mul_ps(vx2, vy2));
	}
	for (s = 0.; i < n; ++i) s += x[i] * y[i];
	_mm_storeu_ps(t, vs1);
	s += t[0] + t[1] + t[2] + t[3];
	_mm_storeu_ps(t, vs2);
	s += t[0] + t[1] + t[2] + t[3];
	return s;
}
static inline void kad_saxpy_inlined(int n, float a, const float *x, float *y) /* BLAS saxpy using SSE */
{
	int i, n8 = n>>3<<3;
	__m128 va;
	va = _mm_set1_ps(a);
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2, vt1, vt2;
		vx1 = _mm_loadu_ps(&x[i]);
		vx2 = _mm_loadu_ps(&x[i+4]);
		vy1 = _mm_loadu_ps(&y[i]);
		vy2 = _mm_loadu_ps(&y[i+4]);
		vt1 = _mm_add_ps(_mm_mul_ps(va, vx1), vy1);
		vt2 = _mm_add_ps(_mm_mul_ps(va, vx2), vy2);
		_mm_storeu_ps(&y[i], vt1);
		_mm_storeu_ps(&y[i+4], vt2);
	}
	for (; i < n; ++i) y[i] += a * x[i];
}
#else
static inline float kad_sdot(int n, const float *x, const float *y) /* BLAS sdot */
{
	int i;
	float s = 0.;
	for (i = 0; i < n; ++i) s += x[i] * y[i];
	return s;
}
static inline void kad_saxpy_inlined(int n, float a, const float *x, float *y) // BLAS saxpy
{
	int i;
	for (i = 0; i < n; ++i) y[i] += a * x[i];
}
#endif

void kad_vec_mul_sum(int n, float *a, const float *b, const float *c)
{
	int i;
	for (i = 0; i < n; ++i) a[i] += b[i] * c[i];
}

void kad_saxpy(int n, float a, const float *x, float *y) { kad_saxpy_inlined(n, a, x, y); }

#ifdef HAVE_CBLAS
#include <cblas.h>
void kad_sgemm_simple(int trans_A, int trans_B, int M, int N, int K, const float *A, const float *B, float *C)
{
	cblas_sgemm(CblasRowMajor, trans_A? CblasTrans : CblasNoTrans, trans_B? CblasTrans : CblasNoTrans, M, N, K, 1.0f, A, trans_A? M : K, B, trans_B? K : N, 1.0f, C, N);
}
#else
void kad_sgemm_simple(int trans_A, int trans_B, int M, int N, int K, const float *A, const float *B, float *C) /* simplified BLAS sgemm */
{
	static const int x = 16;
	int i, j, k;
	if (!trans_A && trans_B) {
		for (i = 0; i < M; i += x)
			for (j = 0; j < N; j += x) {
				int ii, ie = M < i + x? M : i + x;
				int jj, je = N < j + x? N : j + x;
				for (ii = i; ii < ie; ++ii) { /* loop tiling */
					const float *aii = A + ii * K, *bjj;
					float *cii = C + ii * N;
					for (jj = j, bjj = B + j * K; jj < je; ++jj, bjj += K)
						cii[jj] += kad_sdot(K, aii, bjj);
				}
			}
	} else if (!trans_A && !trans_B) {
		for (i = 0; i < M; ++i)
			for (k = 0; k < K; ++k)
				kad_saxpy_inlined(N, A[i*K+k], &B[k*N], &C[i*N]);
	} else if (trans_A && !trans_B) {
		for (k = 0; k < K; ++k)
			for (i = 0; i < M; ++i)
				kad_saxpy_inlined(N, A[k*M+i], &B[k*N], &C[i*N]);
	} else abort(); /* not implemented for (trans_A && trans_B) */
}
#endif

/***************************
 * Random number generator *
 ***************************/

static kad_rng_t kad_rng_dat = { {0x50f5647d2380309dULL, 0x91ffa96fc4c62cceULL}, 0.0, 0, 0 };

static inline uint64_t kad_splitmix64(uint64_t x)
{
	uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static inline uint64_t kad_xoroshiro128plus_next(kad_rng_t *r)
{
	const uint64_t s0 = r->s[0];
	uint64_t s1 = r->s[1];
	const uint64_t result = s0 + s1;
	s1 ^= s0;
	r->s[0] = (s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14);
	r->s[1] = s0 << 36 | s0 >> 28;
	return result;
}

static inline void kad_xoroshiro128plus_jump(kad_rng_t *r)
{
	static const uint64_t JUMP[] = { 0xbeac0467eba5facbULL, 0xd86b048b86aa9922ULL };
	uint64_t s0 = 0, s1 = 0;
	int i, b;
	for (i = 0; i < 2; ++i)
		for (b = 0; b < 64; b++) {
			if (JUMP[i] & 1ULL << b)
				s0 ^= r->s[0], s1 ^= r->s[1];
			kad_xoroshiro128plus_next(r);
		}
	r->s[0] = s0, r->s[1] = s1;
}

void kad_srand(void *d, uint64_t seed)
{
	kad_rng_t *r = d? (kad_rng_t*)d : &kad_rng_dat;
	r->n_gset = 0.0, r->n_iset = 0;
	r->s[0] = kad_splitmix64(seed);
	r->s[1] = kad_splitmix64(r->s[0]);
}

void *kad_rng(void)
{
	kad_rng_t *r;
	r = (kad_rng_t*)calloc(1, sizeof(kad_rng_t));
	kad_xoroshiro128plus_jump(&kad_rng_dat);
	r->s[0] = kad_rng_dat.s[0], r->s[1] = kad_rng_dat.s[1];
	return r;
}

uint64_t kad_rand(void *d) { return kad_xoroshiro128plus_next(d? (kad_rng_t*)d : &kad_rng_dat); }

double kad_drand(void *d)
{
	union { uint64_t i; double d; } u;
	u.i = 0x3FFULL << 52 | kad_xoroshiro128plus_next(d? (kad_rng_t*)d : &kad_rng_dat) >> 12;
	return u.d - 1.0;
}

double kad_drand_normal(void *d)
{
	kad_rng_t *r = d? (kad_rng_t*)d : &kad_rng_dat;
	if (r->n_iset == 0) {
		double fac, rsq, v1, v2;
		do {
			v1 = 2.0 * kad_drand(d) - 1.0;
			v2 = 2.0 * kad_drand(d) - 1.0;
			rsq = v1 * v1 + v2 * v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2.0 * log(rsq) / rsq);
		r->n_gset = v1 * fac;
		r->n_iset = 1;
		return v2 * fac;
	} else {
		r->n_iset = 0;
		return r->n_gset;
	}
}

/*************
 * Operators *
 *************/

static inline void kad_copy_dim1(kad_node_t *dst, const kad_node_t *src) /* set the dimension/shape of dst to src */
{
	dst->n_d = src->n_d;
	if (src->n_d) memcpy(dst->d, src->d, src->n_d * sizeof(int));
}

/********** Arithmetic operations **********/

int kad_op_add(kad_node_t *p, int action)
{
	int i, n0, n1;
	kad_node_t *q[2];

	q[0] = p->child[0], n0 = kad_len(q[0]);
	q[1] = p->child[1], n1 = kad_len(q[1]);
	if (action == KAD_SYNC_DIM) {
		if (n0 % n1 != 0) return -1;
		kad_copy_dim1(p, q[0]);
	} else if (action == KAD_FORWARD) {
		assert(n0 >= n1);
		memcpy(p->x, q[0]->x, n0 * sizeof(float));
		for (i = 0; i < n0; i += n1)
			kad_saxpy(n1, 1.0f, q[1]->x, p->x + i);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0])) kad_saxpy(n0, 1.0f, p->g, q[0]->g);
		if (kad_is_back(q[1]))
			for (i = 0; i < n0; i += n1)
				kad_saxpy(n1, 1.0f, p->g + i, q[1]->g);
	}
	return 0;
}

int kad_op_sub(kad_node_t *p, int action)
{
	int i, n0, n1;
	kad_node_t *q[2];

	q[0] = p->child[0], n0 = kad_len(q[0]);
	q[1] = p->child[1], n1 = kad_len(q[1]);
	if (action == KAD_SYNC_DIM) {
		if (n0 % n1 != 0) return -1;
		kad_copy_dim1(p, q[0]);
	} else if (action == KAD_FORWARD) {
		assert(n0 >= n1);
		memcpy(p->x, q[0]->x, n0 * sizeof(float));
		for (i = 0; i < n0; i += n1)
			kad_saxpy(n1, -1.0f, q[1]->x, p->x + i);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0])) kad_saxpy(n0, 1.0f, p->g, q[0]->g);
		if (kad_is_back(q[1]))
			for (i = 0; i < n0; i += n1)
				kad_saxpy(n1, -1.0f, p->g + i, q[1]->g);
	}
	return 0;
}

int kad_op_mul(kad_node_t *p, int action)
{
	int i, n0, n1;
	kad_node_t *q[2];

	q[0] = p->child[0], n0 = kad_len(q[0]);
	q[1] = p->child[1], n1 = kad_len(q[1]);
	if (action == KAD_SYNC_DIM) {
		if (n0 % n1 != 0) return -1;
		kad_copy_dim1(p, q[0]);
	} else if (action == KAD_FORWARD) {
		assert(n0 >= n1);
		memset(p->x, 0, n0 * sizeof(float));
		if (q[0]->x != 0 && q[1]->x != 0)
			for (i = 0; i < n0; i += n1) /* TODO: optimize when n1==1 */
				kad_vec_mul_sum(n1, p->x + i, q[0]->x + i, q[1]->x);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0]) && q[1]->x)
			for (i = 0; i < n0; i += n1)
				kad_vec_mul_sum(n1, q[0]->g + i, p->g + i, q[1]->x);
		if (kad_is_back(q[1]) && q[0]->x)
			for (i = 0; i < n0; i += n1)
				kad_vec_mul_sum(n1, q[1]->g, p->g + i, q[0]->x + i);
	}
	return 0;
}

int kad_op_cmul(kad_node_t *p, int action)
{
	int i, n_a_row, n_b_row, n_col, n_a_col = 1, n_b_col = 1;
	kad_node_t *q[2];

	q[0] = p->child[0], q[1] = p->child[1];
	n_col = q[0]->d[q[0]->n_d - 1] > q[1]->d[q[1]->n_d - 1]? q[0]->d[q[0]->n_d - 1] : q[1]->d[q[1]->n_d - 1];
	for (i = q[0]->n_d - 1; i >= 0; --i) if (n_a_col < n_col) n_a_col *= q[0]->d[i];
	for (i = q[1]->n_d - 1; i >= 0; --i) if (n_b_col < n_col) n_b_col *= q[1]->d[i];
	n_a_row = kad_len(q[0]) / n_a_col, n_b_row = kad_len(q[1]) / n_b_col;
	if (action == KAD_SYNC_DIM) {
		if (n_a_col != n_b_col) return -1;
		p->n_d = 2, p->d[0] = n_a_row, p->d[1] = n_b_row;
	} else if (action == KAD_FORWARD) {
		memset(p->x, 0, n_a_row * n_b_row * sizeof(float));
		if (q[0]->x && q[1]->x)
			kad_sgemm_simple(0, 1, n_a_row, n_b_row, n_col, q[0]->x, q[1]->x, p->x); /* Y = X * trans(W) */
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0]) && q[1]->x)
			kad_sgemm_simple(0, 0, n_a_row, n_col, n_b_row, p->g, q[1]->x, q[0]->g); /* G_x <- G_y * W */
		if (kad_is_back(q[1]) && q[0]->x)
			kad_sgemm_simple(1, 0, n_b_row, n_col, n_a_row, p->g, q[0]->x, q[1]->g); /* G_w <- trans(G_y) * X */
	}
	return 0;
}

int kad_op_matmul(kad_node_t *p, int action) /* TODO: matmul and cmul have different broadcasting rules */
{
	int n_a_row, n_b_row, n_a_col, n_b_col;
	kad_node_t *q[2];

	q[0] = p->child[0];
	q[1] = p->child[1];
	n_a_row = q[0]->n_d == 1? 1 : q[0]->d[0];
	n_b_row = q[1]->n_d == 1? 1 : q[1]->d[0];
	n_a_col = kad_len(q[0]) / n_a_row;
	n_b_col = kad_len(q[1]) / n_b_row;
	if (action == KAD_SYNC_DIM) {
		if (n_a_col != n_b_row) return -1;
		p->n_d = 2, p->d[0] = n_a_row, p->d[1] = n_b_col;
	} else if (action == KAD_FORWARD) {
		memset(p->x, 0, n_a_row * n_b_col * sizeof(float));
		if (q[0]->x && q[1]->x)
			kad_sgemm_simple(0, 0, n_a_row, n_b_col, n_a_col, q[0]->x, q[1]->x, p->x); /* Y = X * W */
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0]) && q[1]->x)
			kad_sgemm_simple(0, 1, n_a_row, n_a_col, n_b_col, p->g, q[1]->x, q[0]->g); /* G_x <- G_y * trans(W) */
		if (kad_is_back(q[1]) && q[0]->x)
			kad_sgemm_simple(1, 0, n_b_row, n_b_col, n_a_row, q[0]->x, p->g, q[1]->g); /* G_y <- trans(A) * G_y */
	}
	return 0;
}

int kad_op_square(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i)
			p->x[i] = q->x[i] * q->x[i];
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < n; ++i)
			q->g[i] += p->g[i] * (q->x[i] + q->x[i]);
	}
	return 0;
}

int kad_op_1minus(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) p->x[i] = 1.0f - q->x[i];
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		kad_saxpy(n, -1.0f, p->g, q->g);
	}
	return 0;
}

int kad_op_exp(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) p->x[i] = expf(q->x[i]);
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < n; ++i)
			q->g[i] += p->g[i] * p->x[i];
	}
	return 0;
}

int kad_op_log(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) p->x[i] = logf(q->x[i]);
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < n; ++i)
			q->g[i] += p->g[i] / q->x[i];
	}
	return 0;
}

int kad_op_reduce_sum(kad_node_t *p, int action)
{
	kad_node_t *q = p->child[0];
	int i, j, k, axis, d0, d1;

	assert(p->ptr);
	axis = *(int32_t*)p->ptr;
	if (axis < 0 || axis >= q->n_d) return -1;
	for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
	for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
	if (action == KAD_SYNC_DIM) {
		p->n_d = q->n_d - 1;
		for (i = j = 0; i < q->n_d; ++i)
			if (i != axis) p->d[j++] = q->d[i];
	} else if (action == KAD_FORWARD) {
		memset(p->x, 0, kad_len(p) * sizeof(float));
		for (i = 0; i < d0; ++i)
			for (j = 0; j < q->d[axis]; ++j)
				for (k = 0; k < d1; ++k)
					p->x[i * d1 + k] += q->x[(i * q->d[axis] + j) * d1 + k];
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < d0; ++i)
			for (j = 0; j < q->d[axis]; ++j)
				for (k = 0; k < d1; ++k)
					q->g[(i * q->d[axis] + j) * d1 + k] += p->g[i * d1 + k];
	}
	return 0;
}

int kad_op_reduce_mean(kad_node_t *p, int action)
{
	kad_node_t *q = p->child[0];
	int i, j, k, axis, d0, d1;

	assert(p->ptr);
	axis = *(int32_t*)p->ptr;
	if (axis < 0 || axis >= q->n_d) return -1;
	for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
	for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
	if (action == KAD_SYNC_DIM) {
		p->n_d = q->n_d - 1;
		for (i = j = 0; i < q->n_d; ++i)
			if (i != axis) p->d[j++] = q->d[i];
	} else if (action == KAD_FORWARD) {
		float t = 1.0f / q->d[axis];
		memset(p->x, 0, kad_len(p) * sizeof(float));
		for (i = 0; i < d0; ++i)
			for (j = 0; j < q->d[axis]; ++j)
				for (k = 0; k < d1; ++k)
					p->x[i * d1 + k] += t * q->x[(i * q->d[axis] + j) * d1 + k];
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		float t = 1.0f / q->d[axis];
		for (i = 0; i < d0; ++i)
			for (j = 0; j < q->d[axis]; ++j)
				for (k = 0; k < d1; ++k)
					q->g[(i * q->d[axis] + j) * d1 + k] += t * p->g[i * d1 + k];
	}
	return 0;
}

/********** Miscellaneous **********/

int kad_op_dropout(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	assert(p->child[1]->n_d == 0);
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_ALLOC) {
		if (kad_is_back(p->child[0]))
			p->gtmp = realloc(p->gtmp, n);
	} else if (action == KAD_FORWARD) {
		float r = kad_is_const(q) || kad_is_var(q)? 0.0f : *p->child[1]->x, z = 1.0f / (1.0f - r);
		uint8_t *flag = (uint8_t*)p->gtmp;
		for (i = 0; i < n; ++i) {
			int kept = (kad_drand(p->ptr) >= r);
			p->x[i] = kept? q->x[i] * z : 0.0f;
			if (flag) flag[i] = kept;
		}
	} else if (action == KAD_BACKWARD && kad_is_back(p->child[0])) {
		float r = kad_is_const(q) || kad_is_var(q)? 0.0f : *p->child[1]->x, z = 1.0f / (1.0f - r);
		uint8_t *flag = (uint8_t*)p->gtmp;
		for (i = 0; i < n; ++i)
			if (flag[i]) q->g[i] += z * p->g[i];
	}
	return 0;
}

int kad_op_sample_normal(kad_node_t *p, int action) /* not tested */
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_ALLOC) {
		if (kad_is_back(p->child[0]))
			p->gtmp = realloc(p->gtmp, n * sizeof(float));
	} else if (action == KAD_FORWARD) {
		float *r = (float*)p->gtmp;
		for (i = 0; i < n; ++i) {
			float z;
			z = (float)kad_drand_normal(p->ptr);
			p->x[i] = q->x[i] * z;
			if (r) r[i] = z;
		}
	} else if (action == KAD_BACKWARD && kad_is_back(p->child[0])) {
		float *r = (float*)p->gtmp;
		for (i = 0; i < n; ++i)
			q->g[i] += p->g[i] * r[i];
	}
	return 0;
}

int kad_op_slice(kad_node_t *p, int action)
{
	kad_node_t *q = p->child[0];
	int32_t *aux, *range;
	int i, axis, d0, d1;

	assert(p->ptr);
	aux = (int32_t*)p->ptr, axis = aux[0], range = aux + 1;
	if (axis < 0 || axis >= q->n_d) return -1;
	for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
	for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
	if (action == KAD_SYNC_DIM) {
		if (range[0] >= range[1] || range[0] < 0 || range[1] > q->d[axis]) return -1;
		kad_copy_dim1(p, q);
		p->d[axis] = range[1] - range[0];
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < d0; ++i)
			memcpy(&p->x[i * p->d[axis] * d1], &q->x[(i * q->d[axis] + range[0]) * d1], (range[1] - range[0]) * d1 * sizeof(float));
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < d0; ++i)
			kad_saxpy((range[1] - range[0]) * d1, 1.0f, &p->g[i * p->d[axis] * d1], &q->g[(i * q->d[axis] + range[0]) * d1]);
	}
	return 0;
}

int kad_op_concat(kad_node_t *p, int action)
{
	kad_node_t *q = p->child[0];
	int32_t *aux;
	int i, j, k, axis, d0, d1;

	assert(p->ptr);
	aux = (int32_t*)p->ptr, axis = aux[0];
	for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
	for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
	if (action == KAD_SYNC_DIM) {
		for (i = 1; i < p->n_child; ++i) {
			if (p->child[i]->n_d != q->n_d) return -1;
			for (j = 0; j < q->n_d; ++j)
				if (j != axis && q->d[j] != p->child[i]->d[j]) return -1;
		}
		kad_copy_dim1(p, q);
		for (i = 1; i < p->n_child; ++i)
			p->d[axis] += p->child[i]->d[axis];
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < d0; ++i)
			for (j = k = 0; j < p->n_child; ++j) {
				q = p->child[j];
				memcpy(&p->x[(i * p->d[axis] + k) * d1], &q->x[i * q->d[axis] * d1], q->d[axis] * d1 * sizeof(float));
				k += q->d[axis];
			}
	} else if (action == KAD_BACKWARD) {
		for (i = 0; i < d0; ++i)
			for (j = k = 0; j < p->n_child; ++j) {
				q = p->child[j];
				if (!kad_is_back(q)) continue;
				kad_saxpy(q->d[axis] * d1, 1.0f, &p->g[(i * p->d[axis] + k) * d1], &q->g[i * q->d[axis] * d1]);
				k += q->d[axis];
			}
	}
	return 0;
}

int kad_op_reshape(kad_node_t *p, int action)
{
	kad_node_t *q = p->child[0];

	if (action == KAD_SYNC_DIM) {
		if (p->ptr) {
			int32_t *aux = (int32_t*)p->ptr;
			int i, len = 1, n_missing = 0;
			p->n_d = p->ptr_size / 4;
			for (i = 0; i < p->n_d; ++i) p->d[i] = aux[i];
			for (i = 0; i < p->n_d; ++i)
				if (p->d[i] <= 0) ++n_missing;
				else len *= p->d[i];
			if (n_missing == 0 && len != kad_len(q)) return -1;
			if (n_missing > 1) { /* attempt to infer missing dimensions except the last one */
				for (i = 0; i < p->n_d; ++i)
					if (p->d[i] <= 0 && i < q->n_d) {
						p->d[i] = q->d[i], len *= p->d[i];
						if (--n_missing == 1) break;
					}
				if (n_missing > 1) return -1;
			}
			if (n_missing == 1) { /* infer the last missing dimension */
				if (kad_len(q) % len != 0) return -1;
				for (i = 0; i < p->n_d; ++i)
					if (p->d[i] <= 0) p->d[i] = kad_len(q) / len;
			}
		} else kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		memcpy(p->x, q->x, kad_len(p) * sizeof(float));
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		kad_saxpy(kad_len(p), 1.0f, p->g, q->g);
	}
	return 0;
}

int kad_op_reverse(kad_node_t *p, int action)
{
	kad_node_t *q = p->child[0];
	int axis, i, j, n, d0, d1;

	axis = p->ptr? *(int32_t*)p->ptr : 0;
	if (axis < 0) axis += q->n_d;
	assert(axis >= 0 && axis < q->n_d);
	for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
	n = q->d[axis];
	for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < d0; ++i)
			for (j = 0; j < n; ++j)
				memcpy(&p->x[(i * n + n - 1 - j) * d1], &q->x[(i * n + j) * d1], d1 * sizeof(float));
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < d0; ++i)
			for (j = 0; j < n; ++j)
				kad_saxpy(d1, 1.0f, &p->g[(i * n + n - 1 - j) * d1], &q->g[(i * n + j) * d1]);
	}
	return 0;
}

/********** Cost functions **********/

int kad_op_mse(kad_node_t *p, int action)
{
	kad_node_t *y1 = p->child[0]; /* test */
	kad_node_t *y0 = p->child[1]; /* truth */
	int i, n;

	n = kad_len(y0);
	if (action == KAD_SYNC_DIM) {
		if (n != kad_len(y1)) return -1;
		p->n_d = 0;
	} else if (action == KAD_FORWARD) {
		double cost = 0.0;
		for (i = 0; i < n; ++i)
			cost += (y1->x[i] - y0->x[i]) * (y1->x[i] - y0->x[i]);
		p->x[0] = (float)(cost / n);
	} else if (action == KAD_BACKWARD && kad_is_back(y1)) {
		float t = 2.0f * p->g[0] / n;
		for (i = 0; i < n; ++i)
			y1->g[i] += t * (y1->x[i] - y0->x[i]);
	}
	return 0;
}

int kad_op_ce_bin(kad_node_t *p, int action)
{
	static const float tiny = 1e-9f;
	kad_node_t *y1 = p->child[0]; /* test */
	kad_node_t *y0 = p->child[1]; /* truth */
	int i, n;

	n = kad_len(y0);
	if (action == KAD_SYNC_DIM) {
		if (n != kad_len(y1)) return -1;
		p->n_d = 0;
	} else if (action == KAD_FORWARD) {
		double cost = 0.0;
		for (i = 0; i < n; ++i) {
			if (y0->x[i] > 0.0f)
				cost += y0->x[i] * log(y0->x[i] / (y1->x[i] > tiny? y1->x[i] : tiny));
			if (1.0f - y0->x[i] > 0.0f)
				cost += (1.0f - y0->x[i]) * log((1.0f - y0->x[i]) / (1.0f - y1->x[i] > tiny? 1.0f - y1->x[i] : tiny));
		}
		p->x[0] = (float)(cost / n);
	} else if (action == KAD_BACKWARD && kad_is_back(y1)) {
		float t = p->g[0] / n;
		for (i = 0; i < n; ++i) {
			if (y0->x[i] > 0.0f)
				y1->g[i] -= t * y0->x[i] / (y1->x[i] > tiny? y1->x[i] : tiny);
			if (1.0f - y0->x[i] > 0.0f)
				y1->g[i] += t * (1.0f - y0->x[i]) / (1.0f - y1->x[i] > tiny? 1.0f - y1->x[i] : tiny);
		}
	}
	return 0;
}

int kad_op_ce_bin_neg(kad_node_t *p, int action)
{
	static const float tiny = 1e-9f;
	kad_node_t *y1 = p->child[0]; /* test */
	kad_node_t *y0 = p->child[1]; /* truth */
	int i, n;

	n = kad_len(y0);
	if (action == KAD_SYNC_DIM) {
		if (n != kad_len(y1)) return -1;
		p->n_d = 0;
	} else if (action == KAD_FORWARD) {
		double cost = 0.0;
		for (i = 0; i < n; ++i) {
			if (1.0f + y0->x[i] > 0.0f)
				cost += .5f * (1.0f + y0->x[i]) * log((1.0f + y0->x[i]) / (1.0f + y1->x[i] > tiny? 1.0f + y1->x[i] : tiny));
			if (1.0f - y0->x[i] > 0.0f)
				cost += .5f * (1.0f - y0->x[i]) * log((1.0f - y0->x[i]) / (1.0f - y1->x[i] > tiny? 1.0f - y1->x[i] : tiny));
		}
		p->x[0] = (float)(cost / n);
	} else if (action == KAD_BACKWARD && kad_is_back(y1)) {
		float t = p->g[0] / n;
		for (i = 0; i < n; ++i) {
			if (1.0f + y0->x[i] > 0.0f)
				y1->g[i] -= .5f * t * (1.0f + y0->x[i]) / (1.0f + y1->x[i] > tiny? 1.0f + y1->x[i] : tiny);
			if (1.0f - y0->x[i] > 0.0f)
				y1->g[i] += .5f * t * (1.0f - y0->x[i]) / (1.0f - y1->x[i] > tiny? 1.0f - y1->x[i] : tiny);
		}
	}
	return 0;
}

int kad_op_ce_multi(kad_node_t *p, int action)
{
	static const float tiny = 1e-9f;
	kad_node_t *y1 = p->child[0]; /* test */
	kad_node_t *y0 = p->child[1]; /* truth */
	kad_node_t *c = 0;
	int i, j, n1, d0;

	n1 = y0->d[y0->n_d - 1];
	d0 = kad_len(y0) / n1;
	if (p->n_child == 3) {
		c = p->child[2];
		assert(c->n_d == 1 && c->d[0] == n1);
	}
	if (action == KAD_SYNC_DIM) {
		if (kad_len(y0) != kad_len(y1) || y0->d[y0->n_d - 1] != y1->d[y1->n_d - 1]) return -1;
		p->n_d = 0;
	} else if (action == KAD_FORWARD) {
		double cost = 0.0;
		if (c == 0) {
			for (j = 0; j < d0; ++j) {
				float *x1 = &y1->x[j * n1], *x0 = &y0->x[j * n1];
				for (i = 0; i < n1; ++i)
					if (x0[i] > 0.0f)
						cost += x0[i] * log(x0[i] / (x1[i] > tiny? x1[i] : tiny));
			}
		} else {
			for (j = 0; j < d0; ++j) {
				float *x1 = &y1->x[j * n1], *x0 = &y0->x[j * n1];
				for (i = 0; i < n1; ++i)
					if (x0[i] > 0.0f)
						cost += c->x[i] * x0[i] * log(x0[i] / (x1[i] > tiny? x1[i] : tiny));
			}
		}
		p->x[0] = (float)(cost / d0);
	} else if (action == KAD_BACKWARD && kad_is_back(y1)) {
		float t = p->g[0] / d0;
		if (c == 0) {
			for (j = 0; j < d0; ++j) {
				float *g = &y1->g[j * n1], *x1 = &y1->x[j * n1], *x0 = &y0->x[j * n1];
				for (i = 0; i < n1; ++i)
					g[i] -= t * x0[i] / (x1[i] > tiny? x1[i] : tiny);
			}
		} else {
			for (j = 0; j < d0; ++j) {
				float *g = &y1->g[j * n1], *x1 = &y1->x[j * n1], *x0 = &y0->x[j * n1];
				for (i = 0; i < n1; ++i)
					g[i] -= t * c->x[i] * x0[i] / (x1[i] > tiny? x1[i] : tiny);
			}
		}
	}
	return 0;
}

/********** Normalization **********/

int kad_op_stdnorm(kad_node_t *p, int action)
{
	int i, j, n, m;
	kad_node_t *q = p->child[0];
	assert(q->n_d > 0);
	n = q->d[q->n_d - 1];
	m = kad_len(q) / n;
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_ALLOC) {
		p->gtmp = realloc(p->gtmp, m * sizeof(float));
	} else if (action == KAD_FORWARD) {
		float *si = (float*)p->gtmp;
		for (j = 0; j < m; ++j) {
			float *px = &p->x[j * n], *qx = &q->x[j * n];
			float avg, std_inv;
			double s;
			for (i = 0, s = 0.0; i < n; ++i) s += qx[i];
			avg = (float)(s / n);
			for (i = 0; i < n; ++i) px[i] = qx[i] - avg;
			for (i = 0, s = 0.0; i < n; ++i) s += px[i] * px[i];
			std_inv = s == 0.0? 1.0f : (float)(1.0 / sqrt(s / n));
			for (i = 0; i < n; ++i) px[i] *= std_inv;
			si[j] = std_inv;
		}
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		float *si = (float*)p->gtmp;
		for (j = 0; j < m; ++j) {
			float *pg = &p->g[j * n], *qg = &q->g[j * n], *px = &p->x[j * n], std_inv = si[j];
			double s, t;
			for (i = 0, s = t = 0.0; i < n; ++i)
				s += pg[i], t += px[i] * pg[i];
			s /= n, t /= n;
			for (i = 0; i < n; ++i)
				qg[i] += std_inv * (pg[i] - s - px[i] * t);
		}
	}
	return 0;
}

/********** Activation functions **********/

int kad_op_sigm(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i)
			p->x[i] = 1.0f / (1.0f + expf(-q->x[i]));
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < n; ++i)
			q->g[i] += p->g[i] * (p->x[i] * (1.0f - p->x[i]));
	}
	return 0;
}

int kad_op_tanh(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) {
			if (q->x[i] < -20.0f) p->x[i] = -1.0f;
			else {
				float y;
				y = expf(-2.0f * q->x[i]);
				p->x[i] = (1.0f - y) / (1.0f + y);
			}
		}
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < n; ++i)
			q->g[i] += p->g[i] * (1.0f - p->x[i] * p->x[i]);
	}
	return 0;
}

int kad_op_relu(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i)
			p->x[i] = q->x[i] > 0.0f? q->x[i] : 0.0f;
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < n; ++i)
			if (q->x[i] > 0.0f)
				q->g[i] += p->g[i];
	}
	return 0;
}

int kad_op_sin(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) p->x[i] = sinf(q->x[i]);
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < n; ++i)
			q->g[i] += p->g[i] * cosf(q->x[i]);
	}
	return 0;
}

int kad_op_softmax(kad_node_t *p, int action)
{
	int i, j, n1, d0;
	kad_node_t *q = p->child[0];

	n1 = q->d[q->n_d - 1];
	d0 = kad_len(q) / n1;
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (j = 0; j < d0; ++j) {
			float s, max, *x = &q->x[j * n1], *y = &p->x[j * n1];
			for (i = 0, max = -FLT_MAX; i < n1; ++i)
				max = max > x[i]? max : x[i];
			for (i = 0, s = 0.0f; i < n1; ++i) {
				y[i] = expf(x[i] - max);
				s += y[i];
			}
			for (i = 0, s = 1.0f / s; i < n1; ++i) y[i] *= s;
		}
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (j = 0; j < d0; ++j) {
			float s, *g = &p->g[j * n1], *y = &p->x[j * n1], *h = &q->g[j * n1];
			for (i = 0, s = 0.0f; i < n1; ++i)
				s += g[i] * y[i];
			for (i = 0; i < n1; ++i)
				h[i] += y[i] * (g[i] - s);
		}
	}
	return 0;
}

/********** Multi-node pooling **********/

int kad_op_avg(kad_node_t *p, int action)
{
	int i, n;
	float tmp;
	kad_node_t *q;

	assert(p->n_child > 0);
	tmp = 1.0f / p->n_child;
	q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		for (i = 1; i < p->n_child; ++i)
			if (kad_len(p->child[i]) != n) return -1;
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		memcpy(p->x, q->x, n * sizeof(float));
		for (i = 1; i < p->n_child; ++i)
			kad_saxpy(n, 1.0f, p->child[i]->x, p->x);
		for (i = 0; i < n; ++i) p->x[i] *= tmp;
	} else if (action == KAD_BACKWARD) {
		for (i = 0; i < p->n_child; ++i)
			if (kad_is_back(p->child[i]))
				kad_saxpy(n, tmp, p->g, p->child[i]->g);
	}
	return 0;
}

int kad_op_max(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		int *max_j;
		for (i = 1; i < p->n_child; ++i)
			if (kad_len(p->child[i]) != n) return -1;
		kad_copy_dim1(p, q);
		max_j = (int*)calloc(n, sizeof(int));
		p->gtmp = max_j;
	} else if (action == KAD_FORWARD) {
		int j, *max_j = (int*)p->gtmp;
		memset(max_j, 0, n * sizeof(int));
		memcpy(p->x, q->x, n * sizeof(float));
		for (j = 1; j < p->n_child; ++j)
			for (i = 0, q = p->child[j]; i < n; ++i)
				if (q->x[i] > p->x[i]) p->x[i] = q->x[i], max_j[i] = j;
	} else if (action == KAD_BACKWARD) {
		int *max_j = (int*)p->gtmp;
		for (i = 0; i < n; ++i)
			p->child[max_j[i]]->g[i] += p->g[i];
	}
	return 0;
}

int kad_op_stack(kad_node_t *p, int action) /* TODO: allow axis, as in TensorFlow */
{
	int i, n, axis = 0;
	kad_node_t *q;

	assert(p->n_child > 0);
	q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		for (i = 1; i < p->n_child; ++i)
			if (kad_len(p->child[i]) != n) return -1;
		p->n_d = q->n_d + 1;
		for (i = 0; i < axis; ++i) p->d[i] = q->d[i];
		p->d[axis] = p->n_child;
		for (; i < q->n_d; ++i) p->d[i+1] = q->d[i];
	} else if (action == KAD_FORWARD) { /* TODO: doesn't work when axis != 0 */
		for (i = 0; i < p->n_child; ++i)
			memcpy(&p->x[i * n], p->child[i]->x, n * sizeof(float));
	} else if (action == KAD_BACKWARD) {
		for (i = 0; i < p->n_child; ++i)
			if (kad_is_back(p->child[i]))
				kad_saxpy(n, 1.0f, &p->g[i * n], p->child[i]->g);
	}
	return 0;
}

int kad_op_select(kad_node_t *p, int action)
{
	kad_node_t *q;
	int i, n, which;

	which = *(int32_t*)p->ptr;
	if (which < 0) which += p->n_child;
	assert(which >= 0 && which < p->n_child);
	q = p->child[which];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		for (i = 0; i < p->n_child; ++i)
			if (p->child[i]->n_d != q->n_d || kad_len(p->child[i]) != n)
				break;
		if (i < p->n_child) return -1;
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		memcpy(p->x, q->x, n * sizeof(float));
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		kad_saxpy(n, 1.0f, p->g, q->g);
	}
	return 0;
}

/********** 2D convolution **********/

static void conv_rot180(int d0, int d1, float *x) /* rotate/reverse a weight martix */
{
	int i, j;
	for (i = 0; i < d0; ++i) {
		float tmp, *xi = &x[i * d1];
		for (j = 0; j < d1>>1; ++j)
			tmp = xi[j], xi[j] = xi[d1-1-j], xi[d1-1-j] = tmp; 
	}
}

static void conv2d_move_1to3(int d[4], const float *x, float *y) /* convert the NCHW shape to the NHWC shape */
{
	int i, j, k, l;
	for (i = 0; i < d[0]; ++i)
		for (j = 0; j < d[1]; ++j)
			for (k = 0; k < d[2]; ++k) {
				int ik = (i * d[2] + k) * d[3], ijk = ((i * d[1] + j) * d[2] + k) * d[3];
				for (l = 0; l < d[3]; ++l)
					y[(ik + l) * d[1] + j] = x[ijk + l];
			}
}

static void conv2d_add_3to1(int d[4], const float *y, float *x) /* convert the NHWC shape back to NCHW and add to another NCHW-shaped array */
{
	int i, j, k, l;
	for (i = 0; i < d[0]; ++i)
		for (j = 0; j < d[1]; ++j)
			for (k = 0; k < d[2]; ++k) {
				int ik = (i * d[2] + k) * d[3], ijk = ((i * d[1] + j) * d[2] + k) * d[3];
				for (l = 0; l < d[3]; ++l)
					x[ijk + l] += y[(ik + l) * d[1] + j];
			}
}

#define conv_out_size(in_size, aux) (((in_size) - (aux)->kernel_size + (aux)->pad[0] + (aux)->pad[1]) / (aux)->stride + 1)

#define process_row_for(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			const float *xl = &_xx[l - _pad]; \
			for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
			kad_saxpy(_pn, _ww[l], _t, _yy); \
		} \
	} else for (l = 0; l < _wn; ++l) kad_saxpy(_pn, _ww[l], &_xx[l - _pad], _yy); \
} while (0)

#define process_row_back_x(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			float *xl = &_xx[l - _pad]; \
			memset(_t, 0, _pn * sizeof(float)); \
			kad_saxpy(_pn, _ww[l], _yy, _t); \
			for (j = 0; j < _pn; ++j, xl += _stride) *xl += _t[j]; \
		} \
	} else for (l = 0; l < _wn; ++l) kad_saxpy(_pn, _ww[l], _yy, &_xx[l - _pad]); \
} while (0)

#define process_row_back_w(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			const float *xl = &_xx[l - _pad]; \
			for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
			_ww[l] += kad_sdot(_pn, _yy, _t); \
		} \
	} else for (l = 0; l < _wn; ++l) _ww[l] += kad_sdot(_pn, _yy, &_xx[l - _pad]); \
} while (0)

/* Forward and backward passes are implemented with two different algorithms.
 * The first is faster for small kernels with few input channels; otherwise the
 * second algorithm is faster. Both algorithms should produce identical
 * results, up to the precision of "float".
 */
int kad_op_conv2d(kad_node_t *p, int action) /* in the number-channel-height-width (NCHW) shape */
{
#define conv2d_loop1(_x, _w, _y, _tmp, _row_func) do { /* for the NCHW shape */ \
		int n, c1, c0, i, k, ii; \
		for (n = 0; n < q->d[0]; ++n) /* mini-batch */ \
			for (c1 = 0; c1 < w->d[0]; ++c1) /* output channel */ \
				for (c0 = 0; c0 < w->d[1]; ++c0) /* input channel */ \
					for (k = 0; k < w->d[2]; ++k) { /* kernel row */ \
						float *_ww = &(_w)[((c1 * w->d[1] + c0) * w->d[2] + k) * w->d[3]]; \
						for (i = 0, ii = k - aux[0].pad[0]; i < p->d[2] && ii >= 0 && ii < q->d[2]; ++i, ii += aux[0].stride) { /* output row */ \
							float *_xx = &(_x)[((n * q->d[1] + c0) * q->d[2] + ii) * q->d[3]]; \
							float *_yy = &(_y)[((n * p->d[1] + c1) * p->d[2] + i)  * p->d[3]]; \
							if (x_padded) { \
								memcpy(x_padded + aux[1].pad[0], _xx, q->d[3] * sizeof(float)); \
								_xx = x_padded + aux[1].pad[0]; \
							} \
							_row_func(_xx, _ww, _yy, w->d[3], p->d[3], aux[1].stride, aux[1].pad[0], (_tmp)); \
						} /* ~i */ \
					} /* ~k, c0, c1, n */ \
	} while (0)

#define conv2d_loop2(_x, _w, _y, _code) do { /* for the NHWC shape */ \
		int n, c1, i, j, k, ii, j_skip = aux[1].stride * q->d[1], m = w->d[3] * w->d[1]; \
		for (n = 0; n < q->d[0]; ++n) /* mini-batch */ \
			for (c1 = 0; c1 < w->d[0]; ++c1) /* output channel */ \
				for (k = 0; k < w->d[2]; ++k) { /* kernel row */ \
					float *_ww = &(_w)[(c1 * w->d[2] + k) * m]; \
					for (i = 0, ii = k - aux[0].pad[0]; i < p->d[2] && ii >= 0 && ii < q->d[2]; ++i, ii += aux[0].stride) { /* output and input row */ \
						float *_xx = &(_x)[(n * q->d[2] + ii) * q->d[3] * q->d[1]]; \
						float *_yy = &(_y)[((n * p->d[1] + c1) * p->d[2] + i) * p->d[3]]; \
						if (x_padded) { \
							memcpy(x_padded + aux[1].pad[0] * q->d[1], _xx, q->d[3] * q->d[1] * sizeof(float)); \
							_xx = x_padded; \
						} \
						for (j = 0; j < p->d[3]; ++j, _xx += j_skip, ++_yy) _code; /* output and input column */ \
					} /* ~i */ \
				} /* ~k, c1, n */ \
	} while (0)

	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0], *w = p->child[1];
	float *t = 0, *q1 = 0, *w1 = 0, *x_padded = 0;
	int algo_switch = 0;

	if (action == KAD_FORWARD || action == KAD_BACKWARD) { /* allocate working space */
		if (w->d[3] * w->d[1] < 16) {
			t = (float*)malloc(p->d[3] * sizeof(float));
			x_padded = aux[1].pad[0] + aux[1].pad[1] > 0? (float*)calloc(q->d[3] + aux[1].pad[0] + aux[1].pad[1], sizeof(float)) : 0;
		} else {
			q1 = (float*)malloc(kad_len(q) * sizeof(float));
			w1 = (float*)malloc(kad_len(w) * sizeof(float));
			x_padded = aux[1].pad[0] + aux[1].pad[1] > 0? (float*)calloc((q->d[3] + aux[1].pad[0] + aux[1].pad[1]) * q->d[1], sizeof(float)) : 0;
			algo_switch = 1;
		}
	}
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 4 || w->n_d != 4) return -1;
		if (q->d[1] != w->d[1]) return -1; /* unmatched input channels */
		p->n_d = 4;
		p->d[0] = q->d[0], p->d[1] = w->d[0], p->d[2] = conv_out_size(q->d[2], &aux[0]), p->d[3] = conv_out_size(q->d[3], &aux[1]);
	} else if (action == KAD_FORWARD) {
		conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
		memset(p->x, 0, kad_len(p) * sizeof(float));
		if (!algo_switch) { /* this is the first algorithm */
			conv2d_loop1(q->x, w->x, p->x, t, process_row_for);
		} else { /* this is the second algorithm */
			conv2d_move_1to3(q->d, q->x, q1);
			conv2d_move_1to3(w->d, w->x, w1);
			conv2d_loop2(q1, w1, p->x, (*_yy += kad_sdot(m, _ww, _xx)));
		}
		conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(p->child[0])) { /* backprop to the input array */
			conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
			if (!algo_switch) {
				conv2d_loop1(q->g, w->x, p->g, t, process_row_back_x);
			} else {
				memset(q1, 0, kad_len(q) * sizeof(float));
				conv2d_move_1to3(w->d, w->x, w1);
				conv2d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _ww, _xx));
				conv2d_add_3to1(q->d, q1, q->g);
			}
			conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
		}
		if (kad_is_back(p->child[1])) { /* backprop to the weight matrix */
			conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->g);
			if (!algo_switch) {
				conv2d_loop1(q->x, w->g, p->g, t, process_row_back_w);
			} else {
				conv2d_move_1to3(q->d, q->x, q1);
				memset(w1, 0, kad_len(w) * sizeof(float));
				conv2d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _xx, _ww));
				conv2d_add_3to1(w->d, w1, w->g);
			}
			conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->g);
		}
	}
	free(t); free(q1); free(w1); free(x_padded);
	return 0;
}

int kad_op_max2d(kad_node_t *p, int action)
{
	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0];
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 4) return -1;
		p->n_d = 4;
		p->d[0] = q->d[0], p->d[1] = q->d[1], p->d[2] = conv_out_size(q->d[2], &aux[0]), p->d[3] = conv_out_size(q->d[3], &aux[1]);
	} else if (action == KAD_ALLOC) {
		p->gtmp = realloc(p->gtmp, kad_len(p) * sizeof(int));
	} else if (action == KAD_FORWARD) {
		int rest = 1, len, t, i;
		int *f = (int*)p->gtmp;
		len = kad_len(p);
		for (i = 0; i < len; ++i) p->x[i] = -FLT_MAX;
		for (i = 0; i < p->n_d - 2; ++i) rest *= p->d[i];
		for (t = 0; t < rest; ++t) {
			int i, j, k, l, p_row = p->d[p->n_d - 2], p_col = p->d[p->n_d - 1];
			for (i = 0; i < p_row; ++i) {
				int u = (t * p_row + i) * p_col;
				for (k = 0; k < aux[0].kernel_size; ++k) {
					int v, v0, v_end, ii = i * aux[0].stride + k - aux[0].pad[0];
					if (ii < 0 || ii >= q->d[p->n_d - 2]) continue;
					v0 = (t * q->d[p->n_d - 2] + ii) * q->d[p->n_d - 1];
					v_end = v0 + q->d[p->n_d - 1];
					for (l = 0; l < aux[1].kernel_size; ++l)
						for (j = 0, v = v0 + (l > aux[1].pad[0]? l - aux[1].pad[0] : 0); j < p_col && v < v_end; ++j, v += aux[1].stride)
							if (p->x[u + j] < q->x[v])
								p->x[u + j] = q->x[v], f[u + j] = v;
				} /* ~k */
			} /* ~i */
		}
	} else if (action == KAD_BACKWARD) {
		int i, len, *f = (int*)p->gtmp;
		len = kad_len(p);
		for (i = 0; i < len; ++i) q->g[f[i]] += p->g[i];
	}
	return 0;
}

/********** 1D convolution **********/

static void conv1d_move_1to2(int d[3], const float *x, float *y)
{
	int i, j, k;
	for (k = 0; k < d[0]; ++k)
		for (j = 0; j < d[1]; ++j)
			for (i = 0; i < d[2]; ++i)
				y[(k * d[2] + i) * d[1] + j] = x[(k * d[1] + j) * d[2] + i];
}

static void conv1d_add_2to1(int d[3], const float *y, float *x)
{
	int i, j, k;
	for (k = 0; k < d[0]; ++k)
		for (j = 0; j < d[1]; ++j)
			for (i = 0; i < d[2]; ++i)
				x[(k * d[1] + j) * d[2] + i] += y[(k * d[2] + i) * d[1] + j];
}

int kad_op_conv1d(kad_node_t *p, int action) /* in the number-channel-width (NCW) shape */
{
#define conv1d_loop1(_x, _w, _y, _tmp, _row_func) do { /* for the NCW shape */ \
		int n, c1, c0; \
		for (n = 0; n < q->d[0]; ++n) /* mini-batch */ \
			for (c1 = 0; c1 < w->d[0]; ++c1) /* output channel */ \
				for (c0 = 0; c0 < w->d[1]; ++c0) { /* input channel */ \
					float *_ww = &(_w)[(c1 * w->d[1] + c0) * w->d[2]]; \
					float *_xx = &(_x)[(n  * q->d[1] + c0) * q->d[2]]; \
					float *_yy = &(_y)[(n  * p->d[1] + c1) * p->d[2]]; \
					if (x_padded) { \
						memcpy(x_padded + aux->pad[0], _xx, q->d[2] * sizeof(float)); \
						_xx = x_padded + aux->pad[0]; \
					} \
					_row_func(_xx, _ww, _yy, w->d[2], p->d[2], aux->stride, aux->pad[0], (_tmp)); \
				} /* ~c0, c1, n */ \
	} while (0)

#define conv1d_loop2(_x, _w, _y, _code) do { /* for the NWC shape */ \
		int n, c1, j, j_skip = aux->stride * q->d[1], m = w->d[2] * w->d[1]; \
		for (n = 0; n < q->d[0]; ++n) /* mini-batch */ \
			for (c1 = 0; c1 < w->d[0]; ++c1) { /* output channel */ \
				float *_ww = &(_w)[c1 * m]; \
				float *_xx = &(_x)[n * q->d[1] * q->d[2]]; \
				float *_yy = &(_y)[(n * p->d[1] + c1) * p->d[2]]; \
				if (x_padded) { \
					memcpy(x_padded + aux->pad[0] * q->d[1], _xx, q->d[2] * q->d[1] * sizeof(float)); \
					_xx = x_padded; \
				} \
				for (j = 0; j < p->d[2]; ++j, _xx += j_skip, ++_yy) _code; \
			} /* ~c1, n */ \
	} while (0)

	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0], *w = p->child[1];
	float *t = 0, *q1 = 0, *w1 = 0, *x_padded = 0;
	int algo_switch = 0;

	if (action == KAD_FORWARD || action == KAD_BACKWARD) { /* allocate working space */
		if (w->d[2] * w->d[1] < 32) {
			t = (float*)malloc(p->d[2] * sizeof(float));
			x_padded = aux->pad[0] + aux->pad[1] > 0? (float*)calloc(q->d[2] + aux->pad[0] + aux->pad[1], sizeof(float)) : 0;
		} else {
			q1 = (float*)malloc(kad_len(q) * sizeof(float));
			w1 = (float*)malloc(kad_len(w) * sizeof(float));
			x_padded = aux->pad[0] + aux->pad[1] > 0? (float*)calloc((q->d[2] + aux->pad[0] + aux->pad[1]) * q->d[1], sizeof(float)) : 0;
			algo_switch = 1;
		}
	}
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 3 || w->n_d != 3) return -1;
		if (q->d[1] != w->d[1]) return -1; /* unmatched input channels */
		p->n_d = 3;
		p->d[0] = q->d[0], p->d[1] = w->d[0], p->d[2] = conv_out_size(q->d[2], aux);
	} else if (action == KAD_FORWARD) {
		conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
		memset(p->x, 0, kad_len(p) * sizeof(float));
		if (!algo_switch) { /* this is the first algorithm */
			conv1d_loop1(q->x, w->x, p->x, t, process_row_for);
		} else { /* this is the second algorithm */
			conv1d_move_1to2(q->d, q->x, q1);
			conv1d_move_1to2(w->d, w->x, w1);
			conv1d_loop2(q1, w1, p->x, (*_yy += kad_sdot(m, _ww, _xx)));
		}
		conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(p->child[0])) { /* backprop to the input array */
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
			if (!algo_switch) {
				conv1d_loop1(q->g, w->x, p->g, t, process_row_back_x);
			} else {
				memset(q1, 0, kad_len(q) * sizeof(float));
				conv1d_move_1to2(w->d, w->x, w1);
				conv1d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _ww, _xx));
				conv1d_add_2to1(q->d, q1, q->g);
			}
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
		}
		if (kad_is_back(p->child[1])) { /* backprop to the weight matrix */
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->g);
			if (!algo_switch) {
				conv1d_loop1(q->x, w->g, p->g, t, process_row_back_w);
			} else {
				conv1d_move_1to2(q->d, q->x, q1);
				memset(w1, 0, kad_len(w) * sizeof(float));
				conv1d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _xx, _ww));
				conv1d_add_2to1(w->d, w1, w->g);
			}
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->g);
		}
	}
	free(t); free(q1); free(w1); free(x_padded);
	return 0;
}

int kad_op_max1d(kad_node_t *p, int action)
{
	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0];
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 3) return -1;
		p->n_d = 3;
		p->d[0] = q->d[0], p->d[1] = q->d[1], p->d[2] = conv_out_size(q->d[2], aux);
	} else if (action == KAD_ALLOC) {
		p->gtmp = realloc(p->gtmp, kad_len(p) * sizeof(int));
	} else if (action == KAD_FORWARD) {
		int rest = 1, len, t, i;
		int *f = (int*)p->gtmp;
		len = kad_len(p);
		for (i = 0; i < len; ++i) p->x[i] = -FLT_MAX;
		for (i = 0; i < p->n_d - 1; ++i) rest *= p->d[i];
		for (t = 0; t < rest; ++t) {
			int j, l, p_width = p->d[p->n_d - 1];
			int u = t * p_width, v, v0 = t * q->d[p->n_d - 1], v_end = v0 + q->d[p->n_d - 1];
			for (l = 0; l < aux->kernel_size; ++l)
				for (j = 0, v = v0 + (l > aux->pad[0]? l - aux->pad[0] : 0); j < p_width && v < v_end; ++j, v += aux->stride)
					if (p->x[u + j] < q->x[v])
						p->x[u + j] = q->x[v], f[u + j] = v;
		}
	} else if (action == KAD_BACKWARD) {
		int i, len, *f = (int*)p->gtmp;
		len = kad_len(p);
		for (i = 0; i < len; ++i) q->g[f[i]] += p->g[i];
	}
	return 0;
}

int kad_op_avg1d(kad_node_t *p, int action)
{
	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0];
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 3) return -1;
		p->n_d = 3;
		p->d[0] = q->d[0], p->d[1] = q->d[1], p->d[2] = conv_out_size(q->d[2], aux);
	} else if (action == KAD_ALLOC) {
		p->gtmp = realloc(p->gtmp, kad_len(p) * sizeof(int));
	} else if (action == KAD_FORWARD) {
		int rest = 1, len, t, i;
		int *f = (int*)p->gtmp;
		len = kad_len(p);
		for (i = 0; i < len; ++i) p->x[i] = 0.0f, f[i] = 0;
		for (i = 0; i < p->n_d - 1; ++i) rest *= p->d[i];
		for (t = 0; t < rest; ++t) {
			int j, l, p_width = p->d[p->n_d - 1];
			int u = t * p_width, v, v0 = t * q->d[p->n_d - 1], v_end = v0 + q->d[p->n_d - 1];
			for (l = 0; l < aux->kernel_size; ++l)
				for (j = 0, v = v0 + (l > aux->pad[0]? l - aux->pad[0] : 0); j < p_width && v < v_end; ++j, v += aux->stride)
					p->x[u + j] += q->x[v], ++f[u + j];
		}
		for (i = 0; i < len; ++i) p->x[i] /= f[i];
	} else if (action == KAD_BACKWARD) {
		int rest = 1, t, i;
		int *f = (int*)p->gtmp;
		for (i = 0; i < p->n_d - 1; ++i) rest *= p->d[i];
		for (t = 0; t < rest; ++t) {
			int j, l, p_width = p->d[p->n_d - 1];
			int u = t * p_width, v, v0 = t * q->d[p->n_d - 1], v_end = v0 + q->d[p->n_d - 1];
			for (l = 0; l < aux->kernel_size; ++l)
				for (j = 0, v = v0 + (l > aux->pad[0]? l - aux->pad[0] : 0); j < p_width && v < v_end; ++j, v += aux->stride)
					q->g[v] += p->g[u + j] / f[u + j];
		}
	}
	return 0;
}

/********** List of operators **********/

kad_op_f kad_op_list[KAD_MAX_OP] = {
	0,
	kad_op_add,        /* 1:  element-wise addition */
	kad_op_mul,        /* 2:  element-wise multiplication */
	kad_op_cmul,       /* 3:  column multiplication */
	kad_op_ce_bin_neg, /* 4:  binary cross-entropy for (-1,1) */
	kad_op_square,     /* 5:  square */
	kad_op_sigm,       /* 6:  sigmoid */
	kad_op_tanh,       /* 7:  tanh */
	kad_op_relu,       /* 8:  ReLU */
	kad_op_matmul,     /* 9:  matrix multiplication */
	kad_op_avg,        /* 10: general average pooling (not for ConvNet) */
	kad_op_1minus,     /* 11: 1-x */
	kad_op_select,     /* 12: choose between one of the children */
	kad_op_ce_multi,   /* 13: multi-class cross-entropy */
	kad_op_softmax,    /* 14: softmax */
	kad_op_dropout,    /* 15: dropout */
	kad_op_conv2d,     /* 16: 2D convolution */
	kad_op_max2d,      /* 17: 2D max pooling (for 2D ConvNet) */
	kad_op_conv1d,     /* 18: 1D convolution */
	kad_op_max1d,      /* 19: 1D max pooling (for 1D ConvNet) */
	kad_op_slice,      /* 20: slice data at a dimension */
	kad_op_max,        /* 21: general max pooling */
	kad_op_ce_bin,     /* 22: binary cross-entropy for (0,1) */
	kad_op_sub,        /* 23: element-wise subtraction */
	kad_op_sample_normal,  /* 24: sample from a normal distribution */
	kad_op_reduce_sum,     /* 25 */
	kad_op_reduce_mean,    /* 26 */
	kad_op_log,        /* 27: log() */
	kad_op_avg1d,      /* 28: 1D average pooling (for 1D ConvNet) */
	kad_op_mse,        /* 29: mean square error */
	kad_op_reshape,    /* 30 */
	kad_op_concat,     /* 31 */
	kad_op_stdnorm,    /* 32: layer normalization */
	kad_op_exp,        /* 33: exp() */
	kad_op_sin,        /* 34: sin() */
	kad_op_stack,      /* 35: tf.stack, but on the first axis only */
	kad_op_reverse     /* 36: tf.reverse, but on one axis only */
};

char *kad_op_name[KAD_MAX_OP] = {
	0, "add", "mul", "cmul", "ce_bin_neg", "square", "sigm", "tanh", "relu", "matmul", "avg", "1minus", "select", "ce_multi", "softmax",
	"dropout", "conv2d", "max2d", "conv1d", "max1d", "slice", "max", "ce_bin", "sub", "sample_normal", "reduce_sum", "reduce_mean", "log",
	"avg1d", "mse", "reshape", "concat", "stdnorm", "exp", "sin", "stack", "reverse"
};

/**************************
 *** Debugging routines ***
 **************************/

void kad_trap_fe(void)
{
#ifdef __SSE__
	_MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~(_MM_MASK_INVALID | _MM_MASK_DIV_ZERO));
#endif
}

void kad_print_graph(FILE *fp, int n, kad_node_t **v)
{
	int i, j;
	for (i = 0; i < n; ++i) v[i]->tmp = i;
	for (i = 0; i < n; ++i) {
		kad_node_t *p = v[i];
		fprintf(fp, "%d\t%x:%x\t%d\t", i, p->flag, p->ext_flag, p->ext_label);
		if (p->pre) fprintf(fp, "%d\t", p->pre->tmp);
		else fprintf(fp, ".\t");
		fputs("[", fp);
		for (j = 0; j < p->n_d; ++j) {
			if (j) fputc(',', fp);
			fprintf(fp, "%d", p->d[j]);
		}
		fprintf(fp, "]\t");
		if (p->n_child) {
			fprintf(fp, "%s(", kad_op_name[p->op]);
			for (j = 0; j < p->n_child; ++j) {
				if (j) fputc(',', fp);
				fprintf(fp, "$%d", p->child[j]->tmp);
			}
			fprintf(fp, ")");
		} else fprintf(fp, "%s", kad_is_feed(p)? "feed" : kad_is_var(p)? "var" : kad_is_const(p)? "const" : "N/A");
		fputc('\n', fp);
	}
	for (i = 0; i < n; ++i) v[i]->tmp = 0;
}

static void kad_add_delta(int n, kad_node_t **a, float c, float *delta)
{
	int i, k;
	for (i = k = 0; i < n; ++i)
		if (kad_is_var(a[i])) {
			kad_saxpy(kad_len(a[i]), c, &delta[k], a[i]->x);
			k += kad_len(a[i]);
		}
}

void kad_check_grad(int n, kad_node_t **a, int from)
{
	const float eps = 1e-5f, rel = 1e-7f / eps;
	int i, k, n_var;
	float *g0, *delta, f0, f_minus, f_plus, s0, s1, rel_err, p_m_err;
	n_var = kad_size_var(n, a);
	g0 = (float*)calloc(n_var, sizeof(float));
	f0 = *kad_eval_at(n, a, from);
	kad_grad(n, a, from);
	for (i = k = 0; i < n; ++i)
		if (kad_is_var(a[i])) {
			memcpy(&g0[k], a[i]->g, kad_len(a[i]) * sizeof(float));
			k += kad_len(a[i]);
		}
	delta = (float*)calloc(n_var, sizeof(float));
	for (k = 0; k < n_var; ++k) delta[k] = (float)kad_drand(0) * eps;
	kad_add_delta(n, a, 1.0f, delta);
	f_plus = *kad_eval_at(n, a, from);
	kad_add_delta(n, a, -2.0f, delta);
	f_minus = *kad_eval_at(n, a, from);
	kad_add_delta(n, a, 1.0f, delta);
	s0 = kad_sdot(n_var, g0, delta);
	s1 = .5f * (f_plus - f_minus);
	fprintf(stderr, "Gradient check -- %g <=> %g @ %g -- ", s0/eps, s1/eps, f0);
	if (fabs(s1) >= rel * eps) {
		rel_err = fabsf(fabsf(s0) - fabsf(s1)) / (fabsf(s0) + fabsf(s1));
		p_m_err = fabsf(f_plus + f_minus - 2.0f * f0) / fabsf(f_plus - f_minus);
		fprintf(stderr, "rel_err:%g p_m_err:%g -- ", rel_err, p_m_err);
		if (rel_err >= rel && rel_err > p_m_err) fprintf(stderr, "failed\n");
		else fprintf(stderr, "passed\n");
	} else fprintf(stderr, "skipped\n");
	free(delta); free(g0);
}

/*
  The MIT License

  Copyright (c) 2018-2019 Dana-Farber Cancer Institute
                2016-2018 Broad Institute

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef KANN_H
#define KANN_H

#define KANN_VERSION "r549"

#define KANN_F_IN       0x1   /* input */
#define KANN_F_OUT      0x2   /* output */
#define KANN_F_TRUTH    0x4   /* truth output */
#define KANN_F_COST     0x8   /* final cost */

#define KANN_C_CEB      1   /* binary cross-entropy cost, used with sigmoid */
#define KANN_C_CEM      2   /* multi-class cross-entropy cost, used with softmax */
#define KANN_C_CEB_NEG  3   /* binary cross-enytopy-like cost, used with tanh */
#define KANN_C_MSE      4   /* mean square error */

#define KANN_RNN_VAR_H0 0x1 /* take the initial hidden values as variables */
#define KANN_RNN_NORM   0x2 /* apply layer normalization */

typedef struct {
	int n;            /* number of nodes in the computational graph */
	kad_node_t **v;   /* list of nodes */
	float *x, *g, *c; /* collated variable values, gradients and constant values */
	void *mt;         /* auxiliary data for multi-threading; NULL if multi-threading disabled */
} kann_t;

extern int kann_verbose;

#define kann_size_var(a) kad_size_var((a)->n, (a)->v)
#define kann_size_const(a) kad_size_const((a)->n, (a)->v)
#define kann_dim_in(a) kann_feed_dim((a), KANN_F_IN, 0)
#define kann_dim_out(a) kann_feed_dim((a), KANN_F_TRUTH, 0)
#define kann_srand(seed) kad_srand(0, (seed))
#define kann_drand() kad_drand(0)
#define kann_set_batch_size(ann, B) kad_sync_dim((ann)->n, (ann)->v, (B))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Generate a network from a computational graph
 *
 * A network must have at least one scalar cost node (i.e. whose n_d==0). It
 * may optionally contain other cost nodes or output nodes not leading to the
 * primary cost node.
 *
 * @param cost    cost node (must be a scalar, i.e. cost->n_d==0)
 * @param n_rest  number of other nodes without predecessors
 * @param ...     other nodes (of type kad_node_t*) without predecessors
 *
 * @return network on success, or NULL otherwise
 */
kann_t *kann_new(kad_node_t *cost, int n_rest, ...);

/**
 * Unroll an RNN
 *
 * @param a       network
 * @param len     number of unrolls
 *
 * @return an unrolled network, or NULL if the network is not an RNN
 */
kann_t *kann_unroll(kann_t *a, ...);

kann_t *kann_unroll_array(kann_t *a, int *len);
kann_t *kann_clone(kann_t *a, int batch_size);
void kann_delete(kann_t *a);          /* delete a network generated by kann_new() or kann_layer_final() */
void kann_delete_unrolled(kann_t *a); /* delete a network generated by kann_unroll() */

/**
 * Enable/disable multi-threading (requiring pthread)
 *
 * KANN splits a mini-batch to $n_threads mini-mini-batches and puts each of
 * them on one thread. So far, only kann_cost() takes the advantage of
 * multi-threading.
 *
 * @param ann             network
 * @param n_threads       number of threads; <=1 to completely disable multi-threading
 * @param max_batch_size  max mini-batch size; shall no smaller than n_threads
 */
void kann_mt(kann_t *ann, int n_threads, int max_batch_size);

/**
 * Bind float arrays to feed nodes
 *
 * @param a         network
 * @param ext_flag  required external flags
 * @param ext_label required external label
 * @param x         pointers (size equal to the number of matching feed nodes)
 *
 * @return number of matching feed nodes
 */
int kann_feed_bind(kann_t *a, uint32_t ext_flag, int32_t ext_label, float **x);

/**
 * Compute the cost and optionally gradients
 *
 * @param a          network
 * @param cost_label required external label
 * @param cal_grad   whether to compute gradients
 *
 * @return cost
 */
float kann_cost(kann_t *a, int cost_label, int cal_grad);

int kann_eval(kann_t *a, uint32_t ext_flag, int ext_label);
int kann_eval_out(kann_t *a);
int kann_class_error(const kann_t *ann, int *base);

/**
 * Find a node
 *
 * @param a         network
 * @param ext_flag  required external flags; set to 0 to match all flags
 * @param ext_label required external label
 *
 * @return >=0 if found; -1 if not found; -2 if found multiple
 */
int kann_find(const kann_t *a, uint32_t ext_flag, int32_t ext_label);

/**
 * Get the size of a feed node, assuming mini-batch size 1
 *
 * @param a         network
 * @param ext_flag  required external flags
 * @param ext_label required external label
 *
 * @return size>=0; -1 if not found; -2 if found multiple
 */
int kann_feed_dim(const kann_t *a, uint32_t ext_flag, int32_t ext_label);

/**
 * Get an RNN ready for continuous feeding
 *
 * @param a         network
 */
void kann_rnn_start(kann_t *a);

void kann_rnn_end(kann_t *a);

/**
 * Switch between training and prediction networks (effective only when there are switch nodes)
 *
 * @param a         network
 * @param is_train  0 for prediction network and non-zero for training net
 */
void kann_switch(kann_t *a, int is_train);

/**
 * RMSprop update
 *
 * @param n      number of variables
 * @param h0     learning rate
 * @param h      per-variable learning rate; NULL if not applicable
 * @param decay  RMSprop decay; use 0.9 if unsure
 * @param g      gradient, of size n
 * @param t      variables to change
 * @param r      memory, of size n
 */
void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r);

void kann_shuffle(int n, int *s);
float kann_grad_clip(float thres, int n, float *g);

/* common layers */
kad_node_t *kann_layer_input(int n1);
kad_node_t *kann_layer_dense(kad_node_t *in, int n1);
kad_node_t *kann_layer_dropout(kad_node_t *t, float r);
kad_node_t *kann_layer_layernorm(kad_node_t *in);
kad_node_t *kann_layer_rnn(kad_node_t *in, int n1, int rnn_flag);
kad_node_t *kann_layer_lstm(kad_node_t *in, int n1, int rnn_flag);
kad_node_t *kann_layer_gru(kad_node_t *in, int n1, int rnn_flag);
kad_node_t *kann_layer_conv2d(kad_node_t *in, int n_flt, int k_rows, int k_cols, int stride_r, int stride_c, int pad_r, int pad_c);
kad_node_t *kann_layer_conv1d(kad_node_t *in, int n_flt, int k_size, int stride, int pad);
kad_node_t *kann_layer_cost(kad_node_t *t, int n_out, int cost_type);

kad_node_t *kann_new_leaf(uint8_t flag, float x0_01, int n_d, ...); /* flag can be KAD_CONST or KAD_VAR */
kad_node_t *kann_new_scalar(uint8_t flag, float x);
kad_node_t *kann_new_weight(int n_row, int n_col);
kad_node_t *kann_new_bias(int n);
kad_node_t *kann_new_weight_conv2d(int n_out, int n_in, int k_row, int k_col);
kad_node_t *kann_new_weight_conv1d(int n_out, int n_in, int kernel_len);

kad_node_t *kann_new_leaf2(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, ...);
kad_node_t *kann_layer_dense2(int *offset, kad_node_p *par, kad_node_t *in, int n1);
kad_node_t *kann_layer_dropout2(int *offset, kad_node_p *par, kad_node_t *t, float r);
kad_node_t *kann_layer_layernorm2(int *offset, kad_node_t **par, kad_node_t *in);
kad_node_t *kann_layer_rnn2(int *offset, kad_node_t **par, kad_node_t *in, kad_node_t *h0, int rnn_flag);
kad_node_t *kann_layer_gru2(int *offset, kad_node_t **par, kad_node_t *in, kad_node_t *h0, int rnn_flag);

/* operations on network with a single input node and a single output node */
int kann_train_fnn1(kann_t *ann, float lr, int mini_size, int max_epoch, int max_drop_streak, float frac_val, int n, float **_x, float **_y);
float kann_cost_fnn1(kann_t *a, int n, float **x, float **y);
const float *kann_apply1_to(kann_t *a, float *x, int ext_flag, int ext_label);
const float *kann_apply1(kann_t *a, float *x);

/* model I/O */
void kann_save_fp(FILE *fp, kann_t *ann);
void kann_save(const char *fn, kann_t *ann);
kann_t *kann_load_fp(FILE *fp);
kann_t *kann_load(const char *fn);

#ifdef __cplusplus
}
#endif

#endif

#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>

int kann_verbose = 3;

/******************************************
 *** @@BASIC: fundamental KANN routines ***
 ******************************************/

static void kad_ext_collate(int n, kad_node_t **a, float **_x, float **_g, float **_c)
{
	int i, j, k, l, n_var;
	float *x, *g, *c;
	n_var = kad_size_var(n, a);
	x = *_x = (float*)realloc(*_x, n_var * sizeof(float));
	g = *_g = (float*)realloc(*_g, n_var * sizeof(float));
	c = *_c = (float*)realloc(*_c, kad_size_const(n, a) * sizeof(float));
	memset(g, 0, n_var * sizeof(float));
	for (i = j = k = 0; i < n; ++i) {
		kad_node_t *v = a[i];
		if (kad_is_var(v)) {
			l = kad_len(v);
			memcpy(&x[j], v->x, l * sizeof(float));
			free(v->x);
			v->x = &x[j];
			v->g = &g[j];
			j += l;
		} else if (kad_is_const(v)) {
			l = kad_len(v);
			memcpy(&c[k], v->x, l * sizeof(float));
			free(v->x);
			v->x = &c[k];
			k += l;
		}
	}
}

static void kad_ext_sync(int n, kad_node_t **a, float *x, float *g, float *c)
{
	int i, j, k;
	for (i = j = k = 0; i < n; ++i) {
		kad_node_t *v = a[i];
		if (kad_is_var(v)) {
			v->x = &x[j];
			v->g = &g[j];
			j += kad_len(v);
		} else if (kad_is_const(v)) {
			v->x = &c[k];
			k += kad_len(v);
		}
	}
}

kann_t *kann_new(kad_node_t *cost, int n_rest, ...)
{
	kann_t *a;
	int i, n_roots = 1 + n_rest, has_pivot = 0, has_recur = 0;
	kad_node_t **roots;
	va_list ap;

	if (cost->n_d != 0) return 0;

	va_start(ap, n_rest);
	roots = (kad_node_t**)malloc((n_roots + 1) * sizeof(kad_node_t*));
	for (i = 0; i < n_rest; ++i)
		roots[i] = va_arg(ap, kad_node_t*);
	roots[i++] = cost;
	va_end(ap);

	cost->ext_flag |= KANN_F_COST;
	a = (kann_t*)calloc(1, sizeof(kann_t));
	a->v = kad_compile_array(&a->n, n_roots, roots);

	for (i = 0; i < a->n; ++i) {
		if (a->v[i]->pre) has_recur = 1;
		if (kad_is_pivot(a->v[i])) has_pivot = 1;
	}
	if (has_recur && !has_pivot) { /* an RNN that doesn't have a pivot; then add a pivot on top of cost and recompile */
		cost->ext_flag &= ~KANN_F_COST;
		roots[n_roots-1] = cost = kad_avg(1, &cost), cost->ext_flag |= KANN_F_COST;
		free(a->v);
		a->v = kad_compile_array(&a->n, n_roots, roots);
	}
	kad_ext_collate(a->n, a->v, &a->x, &a->g, &a->c);
	free(roots);
	return a;
}

kann_t *kann_clone(kann_t *a, int batch_size)
{
	kann_t *b;
	b = (kann_t*)calloc(1, sizeof(kann_t));
	b->n = a->n;
	b->v = kad_clone(a->n, a->v, batch_size);
	kad_ext_collate(b->n, b->v, &b->x, &b->g, &b->c);
	return b;
}

kann_t *kann_unroll_array(kann_t *a, int *len)
{
	kann_t *b;
	b = (kann_t*)calloc(1, sizeof(kann_t));
	b->x = a->x, b->g = a->g, b->c = a->c; /* these arrays are shared */
	b->v = kad_unroll(a->n, a->v, &b->n, len);
	return b;
}

kann_t *kann_unroll(kann_t *a, ...)
{
	kann_t *b;
	va_list ap;
	int i, n_pivots, *len;
	n_pivots = kad_n_pivots(a->n, a->v);
	len = (int*)calloc(n_pivots, sizeof(int));
	va_start(ap, a);
	for (i = 0; i < n_pivots; ++i) len[i] = va_arg(ap, int);
	va_end(ap);
	b = kann_unroll_array(a, len);
	free(len);
	return b;
}

void kann_delete_unrolled(kann_t *a)
{
	if (a && a->mt) kann_mt(a, 0, 0);
	if (a && a->v) kad_delete(a->n, a->v);
	free(a);
}

void kann_delete(kann_t *a)
{
	if (a == 0) return;
	free(a->x); free(a->g); free(a->c);
	kann_delete_unrolled(a);
}

static void kann_switch_core(kann_t *a, int is_train)
{
	int i;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->op == 12 && a->v[i]->n_child == 2)
			*(int32_t*)a->v[i]->ptr = !!is_train;
}

#define chk_flg(flag, mask) ((mask) == 0 || ((flag) & (mask)))
#define chk_lbl(label, query) ((query) == 0 || (label) == (query))

int kann_find(const kann_t *a, uint32_t ext_flag, int32_t ext_label)
{
	int i, k, r = -1;
	for (i = k = 0; i < a->n; ++i)
		if (chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
			++k, r = i;
	return k == 1? r : k == 0? -1 : -2;
}

int kann_feed_bind(kann_t *a, uint32_t ext_flag, int32_t ext_label, float **x)
{
	int i, k;
	if (x == 0) return 0;
	for (i = k = 0; i < a->n; ++i)
		if (kad_is_feed(a->v[i]) && chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
			a->v[i]->x = x[k++];
	return k;
}

int kann_feed_dim(const kann_t *a, uint32_t ext_flag, int32_t ext_label)
{
	int i, k, n = 0;
	for (i = k = 0; i < a->n; ++i)
		if (kad_is_feed(a->v[i]) && chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
			++k, n = a->v[i]->n_d > 1? kad_len(a->v[i]) / a->v[i]->d[0] : a->v[i]->n_d == 1? a->v[i]->d[0] : 1;
	return k == 1? n : k == 0? -1 : -2;
}

static float kann_cost_core(kann_t *a, int cost_label, int cal_grad)
{
	int i_cost;
	float cost;
	i_cost = kann_find(a, KANN_F_COST, cost_label);
	assert(i_cost >= 0);
	cost = *kad_eval_at(a->n, a->v, i_cost);
	if (cal_grad) kad_grad(a->n, a->v, i_cost);
	return cost;
}

int kann_eval(kann_t *a, uint32_t ext_flag, int ext_label)
{
	int i, k;
	for (i = k = 0; i < a->n; ++i)
		if (chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
			++k, a->v[i]->tmp = 1;
	kad_eval_marked(a->n, a->v);
	return k;
}

void kann_rnn_start(kann_t *a)
{
	int i;
	kann_set_batch_size(a, 1);
	for (i = 0; i < a->n; ++i) {
		kad_node_t *p = a->v[i];
		if (p->pre) { /* NB: BE CAREFUL of the interaction between kann_rnn_start() and kann_set_batch_size() */
			kad_node_t *q = p->pre;
			if (q->x) memcpy(p->x, q->x, kad_len(p) * sizeof(float));
			else memset(p->x, 0, kad_len(p) * sizeof(float));
			if (q->n_child > 0) free(q->x);
			q->x = p->x;
		}
	}
}

void kann_rnn_end(kann_t *a)
{
	int i;
	kad_ext_sync(a->n, a->v, a->x, a->g, a->c);
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->pre && a->v[i]->pre->n_child > 0)
			a->v[i]->pre->x = (float*)calloc(kad_len(a->v[i]->pre), sizeof(float));
}

static int kann_class_error_core(const kann_t *ann, int *base)
{
	int i, j, k, m, n, off, n_err = 0;
	for (i = 0, *base = 0; i < ann->n; ++i) {
		kad_node_t *p = ann->v[i];
		if (((p->op == 13 && (p->n_child == 2 || p->n_child == 3)) || (p->op == 22 && p->n_child == 2)) && p->n_d == 0) { /* ce_bin or ce_multi */
			kad_node_t *x = p->child[0], *t = p->child[1];
			n = t->d[t->n_d - 1], m = kad_len(t) / n;
			for (j = off = 0; j < m; ++j, off += n) {
				float t_sum = 0.0f, t_min = 1.0f, t_max = 0.0f, x_max = 0.0f, x_min = 1.0f;
				int x_max_k = -1, t_max_k = -1;
				for (k = 0; k < n; ++k) {
					float xk = x->x[off+k], tk = t->x[off+k];
					t_sum += tk;
					t_min = t_min < tk? t_min : tk;
					x_min = x_min < xk? x_min : xk;
					if (t_max < tk) t_max = tk, t_max_k = k;
					if (x_max < xk) x_max = xk, x_max_k = k;
				}
				if (t_sum - 1.0f == 0 && t_min >= 0.0f && x_min >= 0.0f && x_max <= 1.0f) {
					++(*base);
					n_err += (x_max_k != t_max_k);
				}
			}
		}
	}
	return n_err;
}

/*************************
 * @@MT: multi-threading *
 *************************/

#ifdef HAVE_PTHREAD
#include <pthread.h>

struct mtaux_t;

typedef struct { /* per-worker data */
	kann_t *a;
	float cost;
	int action;
	pthread_t tid;
	struct mtaux_t *g;
} mtaux1_t;

typedef struct mtaux_t { /* cross-worker data */
	int n_threads, max_batch_size;
	int cal_grad, cost_label, eval_out;
	volatile int n_idle; /* we will be busy waiting on this, so volatile necessary */
	pthread_mutex_t mtx;
	pthread_cond_t cv;
	mtaux1_t *mt;
} mtaux_t;

static void *mt_worker(void *data) /* pthread worker */
{
	mtaux1_t *mt1 = (mtaux1_t*)data;
	mtaux_t *mt = mt1->g;
	for (;;) {
		int action;
		pthread_mutex_lock(&mt->mtx);
		mt1->action = 0;
		++mt->n_idle;
		while (mt1->action == 0)
			pthread_cond_wait(&mt->cv, &mt->mtx);
		action = mt1->action;
		pthread_mutex_unlock(&mt->mtx);
		if (action == -1) break;

		if (mt->eval_out) kann_eval(mt1->a, KANN_F_OUT, 0);
		else mt1->cost = kann_cost_core(mt1->a, mt->cost_label, mt->cal_grad);
	}
	pthread_exit(0);
}

static void mt_destroy(mtaux_t *mt) /* de-allocate an entire mtaux_t struct */
{
	int i;
	pthread_mutex_lock(&mt->mtx);
	mt->n_idle = 0;
	for (i = 1; i < mt->n_threads; ++i) mt->mt[i].action = -1;
	pthread_cond_broadcast(&mt->cv);
	pthread_mutex_unlock(&mt->mtx);
	for (i = 1; i < mt->n_threads; ++i) pthread_join(mt->mt[i].tid, 0);
	for (i = 0; i < mt->n_threads; ++i) kann_delete(mt->mt[i].a);
	free(mt->mt);
	pthread_cond_destroy(&mt->cv);
	pthread_mutex_destroy(&mt->mtx);
	free(mt);
}

void kann_mt(kann_t *ann, int n_threads, int max_batch_size)
{
	mtaux_t *mt;
	int i, k;

	if (n_threads <= 1) {
		if (ann->mt) mt_destroy((mtaux_t*)ann->mt);
		ann->mt = 0;
		return;
	}
	if (n_threads > max_batch_size) n_threads = max_batch_size;
	if (n_threads <= 1) return;

	mt = (mtaux_t*)calloc(1, sizeof(mtaux_t));
	mt->n_threads = n_threads, mt->max_batch_size = max_batch_size;
	pthread_mutex_init(&mt->mtx, 0);
	pthread_cond_init(&mt->cv, 0);
	mt->mt = (mtaux1_t*)calloc(n_threads, sizeof(mtaux1_t));
	for (i = k = 0; i < n_threads; ++i) {
		int size = (max_batch_size - k) / (n_threads - i);
		mt->mt[i].a = kann_clone(ann, size);
		mt->mt[i].g = mt;
		k += size;
	}
	for (i = 1; i < n_threads; ++i)
		pthread_create(&mt->mt[i].tid, 0, mt_worker, &mt->mt[i]);
	while (mt->n_idle < n_threads - 1); /* busy waiting until all threads in sync */
	ann->mt = mt;
}

static void mt_kickoff(kann_t *a, int cost_label, int cal_grad, int eval_out)
{
	mtaux_t *mt = (mtaux_t*)a->mt;
	int i, j, k, B, n_var;

	B = kad_sync_dim(a->n, a->v, -1); /* get the current batch size */
	assert(B <= mt->max_batch_size); /* TODO: can be relaxed */
	n_var = kann_size_var(a);

	pthread_mutex_lock(&mt->mtx);
	mt->cost_label = cost_label, mt->cal_grad = cal_grad, mt->eval_out = eval_out;
	for (i = k = 0; i < mt->n_threads; ++i) {
		int size = (B - k) / (mt->n_threads - i);
		for (j = 0; j < a->n; ++j)
			if (kad_is_feed(a->v[j]))
				mt->mt[i].a->v[j]->x = &a->v[j]->x[k * kad_len(a->v[j]) / a->v[j]->d[0]];
		kad_sync_dim(mt->mt[i].a->n, mt->mt[i].a->v, size); /* TODO: we can point ->x to internal nodes, too */
		k += size;
		memcpy(mt->mt[i].a->x, a->x, n_var * sizeof(float));
		mt->mt[i].action = 1;
	}
	mt->n_idle = 0;
	pthread_cond_broadcast(&mt->cv);
	pthread_mutex_unlock(&mt->mtx);
}

float kann_cost(kann_t *a, int cost_label, int cal_grad)
{
	mtaux_t *mt = (mtaux_t*)a->mt;
	int i, j, B, k, n_var;
	float cost;

	if (mt == 0) return kann_cost_core(a, cost_label, cal_grad);
	B = kad_sync_dim(a->n, a->v, -1); /* get the current batch size */
	n_var = kann_size_var(a);

	mt_kickoff(a, cost_label, cal_grad, 0);
	mt->mt[0].cost = kann_cost_core(mt->mt[0].a, cost_label, cal_grad);
	while (mt->n_idle < mt->n_threads - 1); /* busy waiting until all threads in sync */

	memset(a->g, 0, n_var * sizeof(float)); /* TODO: check if this is necessary when cal_grad is false */
	for (i = k = 0, cost = 0.0f; i < mt->n_threads; ++i) {
		int size = (B - k) / (mt->n_threads - i);
		cost += mt->mt[i].cost * size / B;
		kad_saxpy(n_var, (float)size / B, mt->mt[i].a->g, a->g);
		k += size;
	}
	for (j = 0; j < a->n; ++j) { /* copy values back at recurrent nodes (needed by textgen; TODO: temporary solution) */
		kad_node_t *p = a->v[j];
		if (p->pre && p->n_d >= 2 && p->d[0] == B) {
			for (i = k = 0; i < mt->n_threads; ++i) {
				kad_node_t *q = mt->mt[i].a->v[j];
				memcpy(&p->x[k], q->x, kad_len(q) * sizeof(float));
				k += kad_len(q);
			}
		}
	}
	return cost;
}

int kann_eval_out(kann_t *a)
{
	mtaux_t *mt = (mtaux_t*)a->mt;
	int j, B, n_eval;
	if (mt == 0) return kann_eval(a, KANN_F_OUT, 0);
	B = kad_sync_dim(a->n, a->v, -1); /* get the current batch size */
	mt_kickoff(a, 0, 0, 1);
	n_eval = kann_eval(mt->mt[0].a, KANN_F_OUT, 0);
	while (mt->n_idle < mt->n_threads - 1); /* busy waiting until all threads in sync */
	for (j = 0; j < a->n; ++j) { /* copy output values back */
		kad_node_t *p = a->v[j];
		if (p->ext_flag & KANN_F_OUT) {
			int i, t, k, d0 = p->d[0] / B, d1 = 1; /* for RNN, p->d[0] may equal unroll_len * batch_size */
			assert(p->d[0] % B == 0);
			for (i = 1; i < p->n_d; ++i) d1 *= p->d[i];
			for (i = 0; i < d0; ++i) {
				for (t = k = 0; t < mt->n_threads; ++t) { /* similar to the forward pass of kad_op_concat() */
					kad_node_t *q = mt->mt[t].a->v[j];
					int size = q->d[0] / d0;
					memcpy(&p->x[(i * B + k) * d1], &q->x[i * size * d1], size * d1 * sizeof(float));
					k += size;
				}
			}
		}
	}
	return n_eval;
}

int kann_class_error(const kann_t *ann, int *base)
{
	mtaux_t *mt = (mtaux_t*)ann->mt;
	int i, n_err = 0, b = 0;
	if (mt == 0) return kann_class_error_core(ann, base);
	for (i = 0; i < mt->n_threads; ++i) {
		n_err += kann_class_error_core(mt->mt[i].a, &b);
		*base += b;
	}
	return n_err;
}

void kann_switch(kann_t *ann, int is_train)
{
	mtaux_t *mt = (mtaux_t*)ann->mt;
	int i;
	if (mt == 0) {
		kann_switch_core(ann, is_train);
		return;
	}
	for (i = 0; i < mt->n_threads; ++i)
		kann_switch_core(mt->mt[i].a, is_train);
}
#else
void kann_mt(kann_t *ann, int n_threads, int max_batch_size) {}
float kann_cost(kann_t *a, int cost_label, int cal_grad) { return kann_cost_core(a, cost_label, cal_grad); }
int kann_eval_out(kann_t *a) { return kann_eval(a, KANN_F_OUT, 0); }
int kann_class_error(const kann_t *a, int *base) { return kann_class_error_core(a, base); }
void kann_switch(kann_t *ann, int is_train) { return kann_switch_core(ann, is_train); }
#endif

/***********************
 *** @@IO: model I/O ***
 ***********************/

#define KANN_MAGIC "KAN\1"

void kann_save_fp(FILE *fp, kann_t *ann)
{
	kann_set_batch_size(ann, 1);
	fwrite(KANN_MAGIC, 1, 4, fp);
	kad_save(fp, ann->n, ann->v);
	fwrite(ann->x, sizeof(float), kann_size_var(ann), fp);
	fwrite(ann->c, sizeof(float), kann_size_const(ann), fp);
}

void kann_save(const char *fn, kann_t *ann)
{
	FILE *fp;
	fp = fn && strcmp(fn, "-")? fopen(fn, "wb") : stdout;
	kann_save_fp(fp, ann);
	fclose(fp);
}

kann_t *kann_load_fp(FILE *fp)
{
	char magic[4];
	kann_t *ann;
	int n_var, n_const;

	fread(magic, 1, 4, fp);
	if (strncmp(magic, KANN_MAGIC, 4) != 0) {
		fclose(fp);
		return 0;
	}
	ann = (kann_t*)calloc(1, sizeof(kann_t));
	ann->v = kad_load(fp, &ann->n);
	n_var = kad_size_var(ann->n, ann->v);
	n_const = kad_size_const(ann->n, ann->v);
	ann->x = (float*)malloc(n_var * sizeof(float));
	ann->g = (float*)calloc(n_var, sizeof(float));
	ann->c = (float*)malloc(n_const * sizeof(float));
	fread(ann->x, sizeof(float), n_var, fp);
	fread(ann->c, sizeof(float), n_const, fp);
	kad_ext_sync(ann->n, ann->v, ann->x, ann->g, ann->c);
	return ann;
}

kann_t *kann_load(const char *fn)
{
	FILE *fp;
	kann_t *ann;
	fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
	ann = kann_load_fp(fp);
	fclose(fp);
	return ann;
}

/**********************************************
 *** @@LAYER: layers and model generation ***
 **********************************************/

/********** General but more complex APIs **********/

kad_node_t *kann_new_leaf_array(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, int32_t d[KAD_MAX_DIM])
{
	int i, len, off = offset && par? *offset : -1;
	kad_node_t *p;

	if (off >= 0 && par[off]) return par[(*offset)++];
	p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	p->n_d = n_d, p->flag = flag;
	memcpy(p->d, d, n_d * sizeof(int32_t));
	len = kad_len(p);
	p->x = (float*)calloc(len, sizeof(float));
	if (p->n_d <= 1) {
		for (i = 0; i < len; ++i)
			p->x[i] = x0_01;
	} else {
		double sdev_inv;
		sdev_inv = 1.0 / sqrt((double)len / p->d[0]);
		for (i = 0; i < len; ++i)
			p->x[i] = (float)(kad_drand_normal(0) * sdev_inv);
	}
	if (off >= 0) par[off] = p, ++(*offset);
	return p;
}

kad_node_t *kann_new_leaf2(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, ...)
{
	int32_t i, d[KAD_MAX_DIM];
	va_list ap;
	va_start(ap, n_d); for (i = 0; i < n_d; ++i) d[i] = va_arg(ap, int); va_end(ap);
	return kann_new_leaf_array(offset, par, flag, x0_01, n_d, d);
}

kad_node_t *kann_layer_dense2(int *offset, kad_node_p *par, kad_node_t *in, int n1)
{
	int n0;
	kad_node_t *w, *b;
	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
	b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
	return kad_add(kad_cmul(in, w), b);
}

kad_node_t *kann_layer_dropout2(int *offset, kad_node_p *par, kad_node_t *t, float r)
{
	kad_node_t *x[2], *cr;
	cr = kann_new_leaf2(offset, par, KAD_CONST, r, 0);
	x[0] = t, x[1] = kad_dropout(t, cr);
	return kad_switch(2, x);
}

kad_node_t *kann_layer_layernorm2(int *offset, kad_node_t **par, kad_node_t *in)
{
	int n0;
	kad_node_t *alpha, *beta;
	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	alpha = kann_new_leaf2(offset, par, KAD_VAR, 1.0f, 1, n0);
	beta  = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n0);
	return kad_add(kad_mul(kad_stdnorm(in), alpha), beta);
}

static inline kad_node_t *cmul_norm2(int *offset, kad_node_t **par, kad_node_t *x, kad_node_t *w, int use_norm)
{
	return use_norm? kann_layer_layernorm2(offset, par, kad_cmul(x, w)) : kad_cmul(x, w);
}

kad_node_t *kann_layer_rnn2(int *offset, kad_node_t **par, kad_node_t *in, kad_node_t *h0, int rnn_flag)
{
	int n0, n1 = h0->d[h0->n_d-1], use_norm = !!(rnn_flag & KANN_RNN_NORM);
	kad_node_t *t, *w, *u, *b, *out;

	u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
	b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
	t = cmul_norm2(offset, par, h0, u, use_norm);
	if (in) {
		n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
		w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
		t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
	}
	out = kad_tanh(kad_add(t, b));
	out->pre = h0;
	return out;
}

kad_node_t *kann_layer_gru2(int *offset, kad_node_t **par, kad_node_t *in, kad_node_t *h0, int rnn_flag)
{
	int n0 = 0, n1 = h0->d[h0->n_d-1], use_norm = !!(rnn_flag & KANN_RNN_NORM);
	kad_node_t *t, *r, *z, *w, *u, *b, *s, *out;

	if (in) n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	/* z = sigm(x_t * W_z + h_{t-1} * U_z + b_z) */
	u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
	b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
	t = cmul_norm2(offset, par, h0, u, use_norm);
	if (in) {
		w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
		t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
	}
	z = kad_sigm(kad_add(t, b));
	/* r = sigm(x_t * W_r + h_{t-1} * U_r + b_r) */
	u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
	b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
	t = cmul_norm2(offset, par, h0, u, use_norm);
	if (in) {
		w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
		t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
	}
	r = kad_sigm(kad_add(t, b));
	/* s = tanh(x_t * W_s + (h_{t-1} # r) * U_s + b_s) */
	u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
	b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
	t = cmul_norm2(offset, par, kad_mul(r, h0), u, use_norm);
	if (in) {
		w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
		t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
	}
	s = kad_tanh(kad_add(t, b));
	/* h_t = z # h_{t-1} + (1 - z) # s */
	out = kad_add(kad_mul(kad_1minus(z), s), kad_mul(z, h0));
	out->pre = h0;
	return out;
}

/********** APIs without offset & par **********/

kad_node_t *kann_new_leaf(uint8_t flag, float x0_01, int n_d, ...)
{
	int32_t i, d[KAD_MAX_DIM];
	va_list ap;
	va_start(ap, n_d); for (i = 0; i < n_d; ++i) d[i] = va_arg(ap, int); va_end(ap);
	return kann_new_leaf_array(0, 0, flag, x0_01, n_d, d);
}

kad_node_t *kann_new_scalar(uint8_t flag, float x) { return kann_new_leaf(flag, x, 0); }
kad_node_t *kann_new_weight(int n_row, int n_col) { return kann_new_leaf(KAD_VAR, 0.0f, 2, n_row, n_col); }
kad_node_t *kann_new_vec(int n, float x) { return kann_new_leaf(KAD_VAR, x, 1, n); }
kad_node_t *kann_new_bias(int n) { return kann_new_vec(n, 0.0f); }
kad_node_t *kann_new_weight_conv2d(int n_out, int n_in, int k_row, int k_col) { return kann_new_leaf(KAD_VAR, 0.0f, 4, n_out, n_in, k_row, k_col); }
kad_node_t *kann_new_weight_conv1d(int n_out, int n_in, int kernel_len) { return kann_new_leaf(KAD_VAR, 0.0f, 3, n_out, n_in, kernel_len); }

kad_node_t *kann_layer_input(int n1)
{
	kad_node_t *t;
	t = kad_feed(2, 1, n1), t->ext_flag |= KANN_F_IN;
	return t;
}

kad_node_t *kann_layer_dense(kad_node_t *in, int n1) { return kann_layer_dense2(0, 0, in, n1); }
kad_node_t *kann_layer_dropout(kad_node_t *t, float r) { return kann_layer_dropout2(0, 0, t, r); }
kad_node_t *kann_layer_layernorm(kad_node_t *in) { return kann_layer_layernorm2(0, 0, in); }

kad_node_t *kann_layer_rnn(kad_node_t *in, int n1, int rnn_flag)
{
	kad_node_t *h0;
	h0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
	h0->x = (float*)calloc(n1, sizeof(float));
	return kann_layer_rnn2(0, 0, in, h0, rnn_flag);
}

kad_node_t *kann_layer_gru(kad_node_t *in, int n1, int rnn_flag)
{
	kad_node_t *h0;
	h0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
	h0->x = (float*)calloc(n1, sizeof(float));
	return kann_layer_gru2(0, 0, in, h0, rnn_flag);
}

static kad_node_t *kann_cmul_norm(kad_node_t *x, kad_node_t *w)
{
	return kann_layer_layernorm(kad_cmul(x, w));
}

kad_node_t *kann_layer_lstm(kad_node_t *in, int n1, int rnn_flag)
{
	int n0;
	kad_node_t *i, *f, *o, *g, *w, *u, *b, *h0, *c0, *c, *out;
	kad_node_t *(*cmul)(kad_node_t*, kad_node_t*) = (rnn_flag & KANN_RNN_NORM)? kann_cmul_norm : kad_cmul;

	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	h0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
	h0->x = (float*)calloc(n1, sizeof(float));
	c0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
	c0->x = (float*)calloc(n1, sizeof(float));

	/* i = sigm(x_t * W_i + h_{t-1} * U_i + b_i) */
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	i = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
	/* f = sigm(x_t * W_f + h_{t-1} * U_f + b_f) */
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_vec(n1, 1.0f); /* see Jozefowicz et al on using a large bias */
	f = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
	/* o = sigm(x_t * W_o + h_{t-1} * U_o + b_o) */
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	o = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
	/* g = tanh(x_t * W_g + h_{t-1} * U_g + b_g) */
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	g = kad_tanh(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
	/* c_t = c_{t-1} # f + g # i */
	c = kad_add(kad_mul(f, c0), kad_mul(g, i)); /* can't be kad_mul(c0, f)!!! */
	c->pre = c0;
	/* h_t = tanh(c_t) # o */
	if (rnn_flag & KANN_RNN_NORM) c = kann_layer_layernorm(c); /* see Ba et al (2016) about how to apply layer normalization to LSTM */
	out = kad_mul(kad_tanh(c), o);
	out->pre = h0;
	return out;
}

kad_node_t *kann_layer_conv2d(kad_node_t *in, int n_flt, int k_rows, int k_cols, int stride_r, int stride_c, int pad_r, int pad_c)
{
	kad_node_t *w;
	w = kann_new_weight_conv2d(n_flt, in->d[1], k_rows, k_cols);
	return kad_conv2d(in, w, stride_r, stride_c, pad_r, pad_c);
}

kad_node_t *kann_layer_conv1d(kad_node_t *in, int n_flt, int k_size, int stride, int pad)
{
	kad_node_t *w;
	w = kann_new_weight_conv1d(n_flt, in->d[1], k_size);
	return kad_conv1d(in, w, stride, pad);
}

kad_node_t *kann_layer_cost(kad_node_t *t, int n_out, int cost_type)
{
	kad_node_t *cost = 0, *truth = 0;
	assert(cost_type == KANN_C_CEB || cost_type == KANN_C_CEM || cost_type == KANN_C_CEB_NEG || cost_type == KANN_C_MSE);
	t = kann_layer_dense(t, n_out);
	truth = kad_feed(2, 1, n_out), truth->ext_flag |= KANN_F_TRUTH;
	if (cost_type == KANN_C_MSE) {
		cost = kad_mse(t, truth);
	} else if (cost_type == KANN_C_CEB) {
		t = kad_sigm(t);
		cost = kad_ce_bin(t, truth);
	} else if (cost_type == KANN_C_CEB_NEG) {
		t = kad_tanh(t);
		cost = kad_ce_bin_neg(t, truth);
	} else if (cost_type == KANN_C_CEM) {
		t = kad_softmax(t);
		cost = kad_ce_multi(t, truth);
	}
	t->ext_flag |= KANN_F_OUT, cost->ext_flag |= KANN_F_COST;
	return cost;
}

void kann_shuffle(int n, int *s)
{
	int i, j, t;
	for (i = 0; i < n; ++i) s[i] = i;
	for (i = n; i > 0; --i) {
		j = (int)(i * kad_drand(0));
		t = s[j], s[j] = s[i-1], s[i-1] = t;
	}
}

/***************************
 *** @@MIN: minimization ***
 ***************************/

#ifdef __SSE__
#include <xmmintrin.h>

void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r)
{
	int i, n4 = n>>2<<2;
	__m128 vh, vg, vr, vt, vd, vd1, tmp, vtiny;
	vh = _mm_set1_ps(h0);
	vd = _mm_set1_ps(decay);
	vd1 = _mm_set1_ps(1.0f - decay);
	vtiny = _mm_set1_ps(1e-6f);
	for (i = 0; i < n4; i += 4) {
		vt = _mm_loadu_ps(&t[i]);
		vr = _mm_loadu_ps(&r[i]);
		vg = _mm_loadu_ps(&g[i]);
		if (h) vh = _mm_loadu_ps(&h[i]);
		vr = _mm_add_ps(_mm_mul_ps(vd1, _mm_mul_ps(vg, vg)), _mm_mul_ps(vd, vr));
		_mm_storeu_ps(&r[i], vr);
		tmp = _mm_sub_ps(vt, _mm_mul_ps(_mm_mul_ps(vh, _mm_rsqrt_ps(_mm_add_ps(vtiny, vr))), vg));
		_mm_storeu_ps(&t[i], tmp);
	}
	for (; i < n; ++i) {
		r[i] = (1. - decay) * g[i] * g[i] + decay * r[i];
		t[i] -= (h? h[i] : h0) / sqrtf(1e-6f + r[i]) * g[i];
	}
}
#else
void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r)
{
	int i;
	for (i = 0; i < n; ++i) {
		float lr = h? h[i] : h0;
		r[i] = (1.0f - decay) * g[i] * g[i] + decay * r[i];
		t[i] -= lr / sqrtf(1e-6f + r[i]) * g[i];
	}
}
#endif

float kann_grad_clip(float thres, int n, float *g)
{
	int i;
	double s2 = 0.0;
	for (i = 0; i < n; ++i)
		s2 += g[i] * g[i];
	s2 = sqrt(s2);
	if (s2 > thres)
		for (i = 0, s2 = 1.0 / s2; i < n; ++i)
			g[i] *= (float)s2;
	return (float)s2 / thres;
}

/****************************************************************
 *** @@XY: simpler API for network with a single input/output ***
 ****************************************************************/

int kann_train_fnn1(kann_t *ann, float lr, int mini_size, int max_epoch, int max_drop_streak, float frac_val, int n, float **_x, float **_y)
{
	int i, j, *shuf, n_train, n_val, n_in, n_out, n_var, n_const, drop_streak = 0, min_set = 0;
	float **x, **y, *x1, *y1, *r, min_val_cost = FLT_MAX, *min_x, *min_c;

	n_in = kann_dim_in(ann);
	n_out = kann_dim_out(ann);
	if (n_in < 0 || n_out < 0) return -1;
	n_var = kann_size_var(ann);
	n_const = kann_size_const(ann);
	r = (float*)calloc(n_var, sizeof(float));
	shuf = (int*)malloc(n * sizeof(int));
	x = (float**)malloc(n * sizeof(float*));
	y = (float**)malloc(n * sizeof(float*));
	kann_shuffle(n, shuf);
	for (j = 0; j < n; ++j)
		x[j] = _x[shuf[j]], y[j] = _y[shuf[j]];
	n_val = (int)(n * frac_val);
	n_train = n - n_val;
	min_x = (float*)malloc(n_var * sizeof(float));
	min_c = (float*)malloc(n_const * sizeof(float));

	x1 = (float*)malloc(n_in  * mini_size * sizeof(float));
	y1 = (float*)malloc(n_out * mini_size * sizeof(float));
	kann_feed_bind(ann, KANN_F_IN,    0, &x1);
	kann_feed_bind(ann, KANN_F_TRUTH, 0, &y1);

	for (i = 0; i < max_epoch; ++i) {
		int n_proc = 0, n_train_err = 0, n_val_err = 0, n_train_base = 0, n_val_base = 0;
		double train_cost = 0.0, val_cost = 0.0;
		kann_shuffle(n_train, shuf);
		kann_switch(ann, 1);
		while (n_proc < n_train) {
			int b, c, ms = n_train - n_proc < mini_size? n_train - n_proc : mini_size;
			for (b = 0; b < ms; ++b) {
				memcpy(&x1[b*n_in],  x[shuf[n_proc+b]], n_in  * sizeof(float));
				memcpy(&y1[b*n_out], y[shuf[n_proc+b]], n_out * sizeof(float));
			}
			kann_set_batch_size(ann, ms);
			train_cost += kann_cost(ann, 0, 1) * ms;
			c = kann_class_error(ann, &b);
			n_train_err += c, n_train_base += b;
			kann_RMSprop(n_var, lr, 0, 0.9f, ann->g, ann->x, r);
			n_proc += ms;
		}
		train_cost /= n_train;
		kann_switch(ann, 0);
		n_proc = 0;
		while (n_proc < n_val) {
			int b, c, ms = n_val - n_proc < mini_size? n_val - n_proc : mini_size;
			for (b = 0; b < ms; ++b) {
				memcpy(&x1[b*n_in],  x[n_train+n_proc+b], n_in  * sizeof(float));
				memcpy(&y1[b*n_out], y[n_train+n_proc+b], n_out * sizeof(float));
			}
			kann_set_batch_size(ann, ms);
			val_cost += kann_cost(ann, 0, 0) * ms;
			c = kann_class_error(ann, &b);
			n_val_err += c, n_val_base += b;
			n_proc += ms;
		}
		if (n_val > 0) val_cost /= n_val;
		if (kann_verbose >= 3) {
			fprintf(stderr, "epoch: %d; training cost: %g", i+1, train_cost);
			if (n_train_base) fprintf(stderr, " (class error: %.2f%%)", 100.0f * n_train_err / n_train);
			if (n_val > 0) {
				fprintf(stderr, "; validation cost: %g", val_cost);
				if (n_val_base) fprintf(stderr, " (class error: %.2f%%)", 100.0f * n_val_err / n_val);
			}
			fputc('\n', stderr);
		}
		if (i >= max_drop_streak && n_val > 0) {
			if (val_cost < min_val_cost) {
				min_set = 1;
				memcpy(min_x, ann->x, n_var * sizeof(float));
				memcpy(min_c, ann->c, n_const * sizeof(float));
				drop_streak = 0;
				min_val_cost = (float)val_cost;
			} else if (++drop_streak >= max_drop_streak)
				break;
		}
	}
	if (min_set) {
		memcpy(ann->x, min_x, n_var * sizeof(float));
		memcpy(ann->c, min_c, n_const * sizeof(float));
	}

	free(min_c); free(min_x); free(y1); free(x1); free(y); free(x); free(shuf); free(r);
	return i;
}

float kann_cost_fnn1(kann_t *ann, int n, float **x, float **y)
{
	int n_in, n_out, n_proc = 0, mini_size = 64 < n? 64 : n;
	float *x1, *y1;
	double cost = 0.0;

	n_in = kann_dim_in(ann);
	n_out = kann_dim_out(ann);
	if (n <= 0 || n_in < 0 || n_out < 0) return 0.0;

	x1 = (float*)malloc(n_in  * mini_size * sizeof(float));
	y1 = (float*)malloc(n_out * mini_size * sizeof(float));
	kann_feed_bind(ann, KANN_F_IN,    0, &x1);
	kann_feed_bind(ann, KANN_F_TRUTH, 0, &y1);
	kann_switch(ann, 0);
	while (n_proc < n) {
		int b, ms = n - n_proc < mini_size? n - n_proc : mini_size;
		for (b = 0; b < ms; ++b) {
			memcpy(&x1[b*n_in],  x[n_proc+b], n_in  * sizeof(float));
			memcpy(&y1[b*n_out], y[n_proc+b], n_out * sizeof(float));
		}
		kann_set_batch_size(ann, ms);
		cost += kann_cost(ann, 0, 0) * ms;
		n_proc += ms;
	}
	free(y1); free(x1);
	return (float)(cost / n);
}

const float *kann_apply1_to(kann_t *a, float *x, int ext_flag, int ext_label)
{
	int i_out;
	i_out = kann_find(a, ext_flag, ext_label);
	if (i_out < 0) return 0;
	kann_set_batch_size(a, 1);
	kann_feed_bind(a, KANN_F_IN, 0, &x);
	kad_eval_at(a->n, a->v, i_out);
	return a->v[i_out]->x;
}

const float *kann_apply1(kann_t *a, float *x)
{
	return kann_apply1_to(a, x, KANN_F_OUT, 0);
}


#include <math.h>
#include <stdio.h>
#include <float.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#define VERSION "r490"

typedef struct {
	int len, n_char, n_para, *para_len;
	uint8_t *data, **para;
	int c2i[256];
} tg_data_t;

#define kv_roundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))

uint8_t *tg_read_file(const char *fn, int *_len)
{
	const int buf_len = 0x10000;
	int len = 0, max = 0, l;
	FILE *fp;
	uint8_t *buf, *s = 0;

	fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
	buf = (uint8_t*)malloc(buf_len);
	while ((l = fread(buf, 1, buf_len, fp)) > 0) {
		if (len + l > max) {
			max = len + buf_len;
			kv_roundup32(max);
			s = (uint8_t*)realloc(s, max);
		}
		memcpy(&s[len], buf, l);
		len += l;
	}
	s = (uint8_t*)realloc(s, len);
	*_len = len;
	fclose(fp);
	free(buf);
	return s;
}

tg_data_t *tg_init(const char *fn)
{
	int i, j, st, k;
	tg_data_t *tg;
	tg = (tg_data_t*)calloc(1, sizeof(tg_data_t));
	tg->data = tg_read_file(fn, &tg->len);
	for (i = 0; i < tg->len; ++i)
		tg->c2i[tg->data[i]] = 1;
	for (i = j = 0; i < 256; ++i)
		if (tg->c2i[i] == 0) tg->c2i[i] = -1;
		else tg->c2i[i] = j++;
	tg->n_char = j;
	for (i = 1, st = 0, tg->n_para = 0; i < tg->len; ++i)
		if (tg->data[i] == '\n' && tg->data[i-1] == '\n' && i - st > 1)
			++tg->n_para, st = i + 1;
	if (i - st > 1) ++tg->n_para;
	tg->para = (uint8_t**)calloc(tg->n_para, sizeof(uint8_t*));
	tg->para_len = (int*)calloc(tg->n_para, sizeof(int));
	for (i = 1, st = k = 0; i < tg->len; ++i)
		if (tg->data[i] == '\n' && tg->data[i-1] == '\n' && i - st > 1)
			tg->para[k] = &tg->data[st], tg->para_len[k++] = i - st, st = i + 1;
	if (i - st > 1) tg->para[k] = &tg->data[st], tg->para_len[k++] = i - st;
	for (i = 0; i < tg->len; ++i)
		tg->data[i] = tg->c2i[tg->data[i]];
	return tg;
}

void tg_save(const char *fn, kann_t *ann, const int c2i[256])
{
	FILE *fp;
	fp = fn && strcmp(fn, "-")? fopen(fn, "wb") : stdout;
	kann_save_fp(fp, ann);
	fwrite(c2i, sizeof(int), 256, fp);
	fclose(fp);
}

kann_t *tg_load(const char *fn, int c2i[256])
{
	FILE *fp;
	kann_t *ann;
	fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
	ann = kann_load_fp(fp);
	fread(c2i, sizeof(int), 256, fp);
	fclose(fp);
	return ann;
}

void tg_gen(FILE *fp, kann_t *ann, float temp, int len, const int c2i[256], const char *seed)
{
	int i, c, n_char, i2c[256], i_temp;
	float x[256];
	memset(i2c, 0, 256 * sizeof(int));
	for (i = 0; i < 256; ++i)
		if (c2i[i] >= 0) i2c[c2i[i]] = i;
	n_char = kann_dim_in(ann);
	i_temp = kann_find(ann, 0, -1);
	if (i_temp >= 0) ann->v[i_temp]->x[0] = 1.0f / temp;
	kann_rnn_start(ann);
	for (c = 0; c < ann->n; ++c) {
		kad_node_t *p = ann->v[c];
		if (p->pre) {
			int l = kad_len(p);
			for (i = 0; i < l; ++i)
				p->x[i] = 2.0 * kann_drand() - 1.0;
		}
	}
	if (seed) {
		const char *p;
		for (p = seed; *p; ++p) {
			const float *y;
			float max = -1.0f;
			int max_c = -1;
			c = c2i[(int)*p];
			assert(c >= 0);
			memset(x, 0, n_char * sizeof(float));
			x[c] = 1.0f;
			y = kann_apply1(ann, x);
			for (c = 0; c < n_char; ++c)
				if (max < y[c]) max = y[c], max_c = c;
			c = max_c;
		}
		fprintf(fp, "%s%c", seed, i2c[c]);
	} else c = c2i[(int)' '];
	for (i = 0; i < len; ++i) {
		float s, r;
		const float *y;
		memset(x, 0, n_char * sizeof(float));
		x[c] = 1.0f;
		y = kann_apply1(ann, x);
		r = kann_drand();
		for (c = 0, s = 0.0f; c < n_char; ++c)
			if (s + y[c] >= r) break;
			else s += y[c];
		fputc(i2c[c], fp);
	}
	fputc('\n', fp);
	kann_rnn_end(ann);
	if (i_temp >= 0) ann->v[i_temp]->x[0] = 1.0f;
}

float tg_perplexity(kann_t *ann, const tg_data_t *tg)
{
	const float tiny = 1e-6;
	float x[256], p;
	double loss = 0.0;
	int i;
	kann_rnn_start(ann);
	for (i = 0; i < tg->len - 1; ++i) {
		const float *y;
		memset(x, 0, 256 * sizeof(float));
		x[tg->data[i]] = 1.0f;
		y = kann_apply1(ann, x);
		p = y[tg->data[i+1]];
		loss += logf(p > tiny? p : tiny);
	}
	kann_rnn_end(ann);
	return (float)exp(-loss / (tg->len - 1));
}

int tg_urnn_start(kann_t *ann, int batch_size)
{
	int i, j, n, cnt = 0;
	for (i = 0; i < ann->n; ++i) {
		kad_node_t *p = ann->v[i];
		if (p->pre && p->n_d >= 2 && p->pre->n_d == p->n_d && p->pre->n_child == 0 && kad_len(p)/p->d[0] == kad_len(p->pre)/p->pre->d[0])
			p->pre->flag = 0;
	}
	kann_set_batch_size(ann, batch_size);
	for (i = 0; i < ann->n; ++i) {
		kad_node_t *p = ann->v[i];
		if (p->pre && p->n_d >= 2 && p->pre->n_d == p->n_d && p->pre->n_child == 0 && kad_len(p) == kad_len(p->pre)) {
			kad_node_t *q = p->pre;
			n = kad_len(p) / p->d[0];
			memset(p->x, 0, p->d[0] * n * sizeof(float));
			if (q->x)
				for (j = 0; j < p->d[0]; ++j)
					memcpy(&p->x[j * n], q->x, n * sizeof(float));
			q->x = p->x;
			++cnt;
		}
	}
	return cnt;
}

void tg_train(kann_t *ann, const tg_data_t *tg, float lr, int ulen, int vlen, int cs, int mbs, int max_epoch, float grad_clip, const char *fn, int batch_len, int n_threads)
{
	int i, epoch, u, n_var, n_char;
	float **x, **y, *r;
	const uint8_t **p;
	kann_t *ua;

	batch_len = batch_len < tg->len? batch_len : tg->len;
	n_char = kann_dim_in(ann);
	x = (float**)calloc(ulen, sizeof(float*));
	y = (float**)calloc(ulen, sizeof(float*));
	for (u = 0; u < ulen; ++u) {
		x[u] = (float*)calloc(n_char * mbs, sizeof(float));
		y[u] = (float*)calloc(n_char * mbs, sizeof(float));
	}
	n_var = kann_size_var(ann);
	r = (float*)calloc(n_var, sizeof(float));
	p = (const uint8_t**)calloc(mbs, sizeof(const uint8_t*));

	ua = kann_unroll(ann, ulen);
	tg_urnn_start(ua, mbs);
	kann_mt(ua, n_threads, mbs);
	kann_switch(ua, 1);
	kann_feed_bind(ua, KANN_F_IN,  100, x);
	kann_feed_bind(ua, KANN_F_TRUTH, 0, y);
	for (epoch = 0; epoch < max_epoch; ++epoch) {
		double cost = 0.0;
		int c, j, b, tot = 0, ctot = 0, n_cerr = 0;
		for (i = 0; i < batch_len; i += mbs * cs * ulen) {
			for (b = 0; b < mbs; ++b)
				p[b] = tg->data + (int)((tg->len - ulen * cs - 1) * kad_drand(0)) + 1;
			for (j = 0; j < ua->n; ++j) // reset initial hidden values to zero
				if (ua->v[j]->pre)
					memset(ua->v[j]->x, 0, kad_len(ua->v[j]) * sizeof(float));
			for (c = 0; c < cs; ++c) {
				int ce_len = c? ulen : ulen - vlen;
				for (u = 0; u < ulen; ++u) {
					memset(x[u], 0, mbs * n_char * sizeof(float));
					memset(y[u], 0, mbs * n_char * sizeof(float));
				}
				for (b = 0; b < mbs; ++b) {
					for (u = 0; u < ulen; ++u) {
						x[u][b * n_char + p[b][u-1]] = 1.0f;
						if (c || u >= vlen)
							y[u][b * n_char + p[b][u]] = 1.0f;
					}
					p[b] += ulen;
				}
				cost += kann_cost(ua, 0, 1) * ulen * mbs;
				n_cerr += kann_class_error(ua, &b);
				tot += ce_len * mbs, ctot += b;
				if (grad_clip > 0.0f) kann_grad_clip(grad_clip, n_var, ua->g);
				kann_RMSprop(n_var, lr, 0, 0.9f, ua->g, ua->x, r);
			}
		}
		fprintf(stderr, "epoch: %d; running cost: %g (class error: %.2f%%)\n", epoch+1, cost / tot, 100.0 * n_cerr / ctot);
		tg_gen(stderr, ann, 0.4f, 100, tg->c2i, "is");
		if (fn) tg_save(fn, ann, tg->c2i);
	}
	kann_delete_unrolled(ua);

	for (u = 0; u < ulen; ++u) {
		free(x[u]); free(y[u]);
	}
	free(r); free(y); free(x); free(p);
}

static kann_t *model_gen(int model, int n_char, int n_h_layers, int n_h_neurons, float h_dropout, int use_norm)
{
	int i, flag = use_norm? KANN_RNN_NORM : 0;
	kad_node_t *t, *t1;
	t = kann_layer_input(n_char), t->ext_label = 100;
	for (i = 0; i < n_h_layers; ++i) {
		if (model == 0) t = kann_layer_rnn(t, n_h_neurons, flag);
		else if (model == 1) t = kann_layer_lstm(t, n_h_neurons, flag);
		else if (model == 2) t = kann_layer_gru(t, n_h_neurons, flag);
		t = kann_layer_dropout(t, h_dropout);
	}
	t = kann_layer_dense(t, n_char);
	t1 = kann_new_scalar(KAD_CONST, 1.0f), t1->ext_label = -1; // -1 is for backward compatibility
	t = kad_mul(t, t1); // t1 is the inverse of temperature
	t = kad_softmax(t), t->ext_flag |= KANN_F_OUT;
	t1 = kad_feed(2, 1, n_char), t1->ext_flag |= KANN_F_TRUTH;
	t = kad_ce_multi(t, t1), t->ext_flag |= KANN_F_COST;
	return kann_new(t, 0);
}

int main(int argc, char *argv[])
{
	int c, seed = 11, ulen = 70, vlen = 10, n_h_layers = 1, n_h_neurons = 128, model = 2, max_epoch = 50, mbs = 64, c2i[256];
	int len_gen = 1000, use_norm = 1, batch_len = 1000000, n_threads = 1, cal_perp = 0, cs = 100;
	float h_dropout = 0.0f, temp = 0.5f, lr = 0.01f, grad_clip = 10.0f;
	kann_t *ann = 0;
	char *fn_in = 0, *fn_out = 0, *prefix = 0;

	while ((c = getopt(argc, argv, "n:l:s:r:m:B:o:i:d:b:T:M:u:L:g:Np:t:xv:c:")) >= 0) {
		if (c == 'n') n_h_neurons = atoi(optarg);
		else if (c == 'l') n_h_layers = atoi(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'r') lr = atof(optarg);
		else if (c == 'm') max_epoch = atoi(optarg);
		else if (c == 'B') mbs = atoi(optarg);
		else if (c == 'd') h_dropout = atof(optarg);
		else if (c == 'T') temp = atof(optarg);
		else if (c == 'c') cs = atoi(optarg);
		else if (c == 'u') ulen = atoi(optarg);
		else if (c == 'v') vlen = atoi(optarg);
		else if (c == 'L') len_gen = atoi(optarg);
		else if (c == 'g') grad_clip = atof(optarg);
		else if (c == 'N') use_norm = 0;
		else if (c == 'p') prefix = optarg;
		else if (c == 'b') batch_len = atoi(optarg);
		else if (c == 't') n_threads = atoi(optarg);
		else if (c == 'x') cal_perp = 1;
		else if (c == 'M') {
			if (strcmp(optarg, "rnn") == 0) model = 0;
			else if (strcmp(optarg, "lstm") == 0) model = 1;
			else if (strcmp(optarg, "gru") == 0) model = 2;
		}
	}
	if (vlen >= ulen) vlen = ulen - 1;
	if (argc == optind && fn_in == 0) {
		FILE *fp = stdout;
		fprintf(fp, "Usage: textgen [options] <in.txt>\n");
		fprintf(fp, "Options:\n");
		fprintf(fp, "  Model construction:\n");
		fprintf(fp, "    -i FILE     read trained model from FILE []\n");
		fprintf(fp, "    -o FILE     save trained model to FILE []\n");
		fprintf(fp, "    -s INT      random seed [%d]\n", seed);
		fprintf(fp, "    -l INT      number of hidden layers [%d]\n", n_h_layers);
		fprintf(fp, "    -n INT      number of hidden neurons per layer [%d]\n", n_h_neurons);
		fprintf(fp, "    -M STR      model: rnn, lstm or gru [gru]\n");
		fprintf(fp, "    -N          don't use layer normalization\n");
		fprintf(fp, "  Model training:\n");
		fprintf(fp, "    -r FLOAT    learning rate [%g]\n", lr);
		fprintf(fp, "    -d FLOAT    dropout at the hidden layer(s) [%g]\n", h_dropout);
		fprintf(fp, "    -m INT      max number of epochs [%d]\n", max_epoch);
		fprintf(fp, "    -B INT      mini-batch size [%d]\n", mbs);
		fprintf(fp, "    -u INT      max unroll [%d]\n", ulen);
		fprintf(fp, "    -v INT      burn-in length [%d]\n", vlen);
		fprintf(fp, "    -g FLOAT    gradient clipping threshold [%g]\n", grad_clip);
		fprintf(fp, "    -c INT      size of a batch [%d]\n", batch_len);
		fprintf(fp, "    -b          use minibatch (run faster but converge slower)\n");
		fprintf(fp, "    -x          compute perplexity at the end\n");
		fprintf(fp, "  Text generation:\n");
		fprintf(fp, "    -p STR      prefix []\n");
		fprintf(fp, "    -T FLOAT    temperature [%g]\n", temp);
		fprintf(fp, "    -L INT      length of text to generate [%d]\n", len_gen);
		return 1;
	}

	fprintf(stderr, "Version: %s\n", VERSION);
	fprintf(stderr, "Command line:");
	for (c = 0; c < argc; ++c)
		fprintf(stderr, " %s", argv[c]);
	fprintf(stderr, "\n");
	kann_srand(seed);
	kad_trap_fe();
	if (fn_in) ann = tg_load(fn_in, c2i);

	if (argc - optind >= 1) { // train
		tg_data_t *tg;
		tg = tg_init(argv[optind]);
		fprintf(stderr, "Read %d paragraphs and %d characters; alphabet size %d\n", tg->n_para, tg->len, tg->n_char);
		if (!ann) ann = model_gen(model, tg->n_char, n_h_layers, n_h_neurons, h_dropout, use_norm);
		tg_train(ann, tg, lr, ulen, vlen, cs, mbs, max_epoch, grad_clip, fn_out, batch_len, n_threads);
		if (cal_perp) fprintf(stderr, "Character-level perplexity: %g\n", tg_perplexity(ann, tg));
		free(tg->data); free(tg);
	} else tg_gen(stdout, ann, temp, len_gen, c2i, prefix);

	kann_delete(ann);
	return 0;
}
