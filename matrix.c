#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if( rows <= 0 || cols <= 0 ) {
        PyErr_SetString(PyExc_ValueError, "Number of rows and columns must be positive");
        return -1;
    }

    *mat = (matrix*) malloc( sizeof(matrix) );
    if( *mat == NULL ) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix struct");
        return -1;
    }
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->is_1d = rows == 1 || cols == 1 ? 1 : 0;
    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL;

    size_t ptr_bytes = rows * sizeof(double*);
    size_t data_bytes = rows * cols * sizeof(double);
    // Allocate a single block for both row pointers and data
    void *block = calloc(1, ptr_bytes + data_bytes);
    if( block == NULL ) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix data");
        free(*mat);
        *mat = NULL;
        return -1;
    }

    (*mat)->data = (double**) block;
    // Set row pointers    
    double *data_start = (double*) ( (char*)block + ptr_bytes );
    for( size_t i = 0; i < rows; i++ ) {
        (*mat)->data[i] = data_start + i * cols;
    }

    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if( row_offset < 0 || col_offset < 0 || rows <= 0 || cols <= 0 ) {
        PyErr_SetString(PyExc_ValueError, "Invalid slice parameters");
        return -1;
    }
    if( row_offset + rows > from->rows || col_offset + cols > from->cols ) {
        PyErr_SetString(PyExc_ValueError, "Slice parameters out of bounds");
        return -1;
    }
    
    *mat = (matrix*) malloc( sizeof(matrix) );
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->is_1d = (rows == 1 || cols == 1) ? 1 : 0;
    (*mat)->ref_cnt = 1;
    (*mat)->parent = from;
    
    // set row pointers
    (*mat)->data = malloc( rows * sizeof(double*) );
    if( (*mat)->data == NULL ) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for matrix data");
        free( *mat );
        *mat = NULL;
        return -1;
    }
    (*mat)->data = from->data + row_offset;
    for( size_t i = 0; i < rows; i++ ) {
        (*mat)->data[i] = from->data[row_offset + i] + col_offset;
    }

    from->ref_cnt += 1;

    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if( mat == NULL ) {
        return;
    }

    mat->ref_cnt--;
    if( mat->ref_cnt > 0 ) return;

    if( mat->parent != NULL ) {
        // recursively deallocate parent if needed
        deallocate_matrix( mat->parent );
        free( mat );
    } else {
        free( mat->data );
        free( mat );
    }
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    return mat->data[row][col];
}   

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            set(mat, i, j, val);
        }
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if( result->rows != mat1->rows || result->cols != mat1->cols ||
        result->rows != mat2->rows || result->cols != mat2->cols ) {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions must agree for addition");
        return -1;
    }

    for( int i = 0; i < result->rows; i++ ) {
        for( int j = 0; j < result->cols; j++ ) {
            double sum = get(mat1, i, j) + get(mat2, i, j);
            set(result, i, j, sum);
        }
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if( result->rows != mat1->rows || result->cols != mat1->cols ||
        result->rows != mat2->rows || result->cols != mat2->cols ) {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions must agree for subtraction");
        return -1;
    }

    for( int i = 0; i < result->rows; i++ ) {
        for( int j = 0; j < result->cols; j++ ) {
            double diff = get(mat1, i, j) - get(mat2, i, j);
            set(result, i, j, diff);
        }
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if( mat1->cols != mat2->rows ||
        result->rows != mat1->rows || result->cols != mat2->cols ) {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions must agree for multiplication");
        return -1;
    }

    for( int i = 0; i < result->rows; i++ ) {
        for( int j = 0; j < result->cols; j++ ) {
            double sum = 0.0;
            for( int k = 0; k < mat1->cols; k++ ) {
                sum += get(mat1, i, k) * get(mat2, k, j);
            }
            set(result, i, j, sum);
        }
    }
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    if( mat->rows != mat->cols ||
        result->rows != mat->rows || result->cols != mat->cols ||
        pow < 0 ) {
        PyErr_SetString(PyExc_ValueError, "Matrix must be square and power non-negative for exponentiation");
        return -1;
    }

    for( int i = 0; i < result->rows; i++ ) {
        for( int j = 0; j < result->cols; j++ ) {
            if( i == j ) {
                set( result, i, j, 1.0 );
            } else {
                set( result, i, j, 0.0 );
            }
        }
    }

    for( int i = 0; i < pow; i++ ) {
        matrix *temp;
        allocate_matrix(&temp, mat->rows, mat->cols);
        mul_matrix( temp, result, mat );
        // copy temp to result
        for( int r = 0; r < result->rows; r++ ) {
            for( int c = 0; c < result->cols; c++ ) {
                set( result, r, c, get( temp, r, c ) );
            }
        }
        deallocate_matrix( temp );
    }
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if( result->rows != mat->rows || result->cols != mat->cols ) {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions must agree for negation");
        return -1;
    }

    for( int i = 0; i < result->rows; i++ ) {
        for( int j = 0; j < result->cols; j++ ) {
            double neg = -get(mat, i, j);
            set(result, i, j, neg);
        }
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if( result->rows != mat->rows || result->cols != mat->cols ) {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions must agree for absolute value");
        return -1;
    }

    for(int i = 0; i < result->rows; i++ ) {
        for( int j = 0; j < result->cols; j++ ) {
            double val = get(mat, i, j);
            if( val < 0 ) {
                val = -val;
            }
            set(result, i, j, val);
        }
    }
    return 0;
}

