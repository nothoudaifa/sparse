#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// =========================================================
// FUNCTION PROTOTYPE
// =========================================================
void sparse__multiply(
    int rows,
    int cols,
    const double* A,
    const double* x,
    int* out_nnz,
    double* values,
    int* col_indices,
    int* row_ptrs,
    double* y
);

// =========================================================
// TODO: USER IMPLEMENTATION
// =========================================================
void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
) {
    // First extract the sparse matrix into CSR format
    // we have values, col_indices, row_ptrs preallocated so we just need to put values in them


    // CSR_index so we now our index in values and col_indices
    size_t CSR_index = 0;
    // the first entry of row_ptrs is always 0 since we start with the first element the matrix
    // if the first line is empty then row_ptrs[1] == 0 also, that means there is no columns in the line, and so on for each line
    row_ptrs[0] = 0;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            // checking that the cell is not 0
            if (A[i*cols + j] != (double)0) {
                // we write the index and the value in the specific arrays then increment CSR_index
                values[CSR_index] = A[i*cols + j];
                col_indices[CSR_index] = j;
                CSR_index++;
            }
        }
        // we only affect the end of row value since the start of this row is the end of the row before it
        row_ptrs[i+1] = CSR_index;
    }


    // now after getting the CSR matrix we calculate A*x using it and put it in y


    for (size_t i = 0; i < rows; i++) {
        // absolutely need to do this, i spent 30 minutes debugging because i thought it was allocated using calloc (which zeros memory) instead of calloc
        y[i] = 0.0;
        for (size_t j = row_ptrs[i]; j < row_ptrs[i+1]; j++) {
            // we loop through each row in row_ptrs and mutlipy values[j] by x[col_indices[j]]
            // values[j] is the value of the column, but j is not the index of the column, the index is stored in col_indices
            y[i] += values[j] * x[col_indices[j]];
        }
        // if the result is zero we increment out_nnz
        if (y[i] == 0) {
            (*out_nnz)++;
        }
    }
}

// =========================================================
// TEST HARNESS
// =========================================================
int main(void) {
    srand(time(NULL));
    
    const int num_iterations = 100;
    int passed_count = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        int rows = rand() % 41 + 5;
        int cols = rand() % 41 + 5;
        double density = 0.05 + (rand() / (double) RAND_MAX) * 0.35;
        
        size_t mat_sz = (size_t) rows * cols;

        double* A = calloc(mat_sz, sizeof(double));
        for (size_t i = 0; i < mat_sz; ++i) {
            if (((double) rand() / RAND_MAX) < density) {
                A[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
            }
        }

        double* values = malloc(mat_sz * sizeof(double));
        int* col_indices = malloc(mat_sz * sizeof(int));
        int* row_ptrs = malloc((rows + 1) * sizeof(int));
        double* x = malloc(cols * sizeof(double));
        double* y_user = malloc(rows * sizeof(double));
        double* y_ref = calloc(rows, sizeof(double));
        int out_nnz = 0;

        for (int i = 0; i < cols; ++i) {
            x[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
        }

        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int j = 0; j < cols; ++j) {
                sum += A[i * cols + j] * x[j];
            }
            y_ref[i] = sum;
        }


        sparse_multiply(rows, cols, A, x, &out_nnz, values, col_indices, row_ptrs, y_user);
        
        double max_err = 0.0;
        int passed = 1;
        for (int i = 0; i < rows; ++i) {
            double diff = fabs(y_user[i] - y_ref[i]);
            double tol = 1e-7 + 1e-7 * fabs(y_ref[i]); // Mixed absolute/relative tolerance
            if (diff > tol) {
                max_err = fmax(max_err, diff);
                passed = 0;
            }
        }

        if (passed) {
            passed_count++;
        } 

        printf(
            "Iter %2d [%3dx%3d, density=%.2f, nnz=%4d]: %s (Max error: %.2e)\n",
            iter, rows, cols, density, out_nnz, passed ? "PASS" : "FAIL", max_err
        );
        free(A);
        free(values);
        free(col_indices);
        free(row_ptrs);
        free(x);
        free(y_user);
        free(y_ref);
    }

    printf(
        "\n%s (%d/%d iterations passed)\n",
        passed_count == num_iterations ? "All tests passed!" : "Some tests failed.",
        passed_count, num_iterations
    );
           
    return passed_count == num_iterations ? 0 : 1;
}
