package utils

import (
    "gonum.org/v1/gonum/mat"
    "fmt"
    "math"
    )

func Min(a, b int) int {
    if a < b {
        return a
    }

    return b
}


func MatPrint(X mat.Matrix) {
    fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
    fmt.Printf("%v\n", fa)
}


type Activation struct{
    Function, Derivative func(units *mat.Dense)
}

func SigmoidF(units *mat.Dense) {
    size_out, size_in := units.Dims()

    for r := 0; r < size_in; r++{
        for c := 0; c < size_out; c++{
            v := units.At(r, c)
            units.Set(r, c, 1 /(1 + math.Exp(-1 * v)))
        }
    }
}

func SigmoidDeriv(units *mat.Dense){
    var units_minus *mat.Dense

    units_minus.Apply(func (i, j int, v float64) float64 {return 1 - v}, units_minus)
    SigmoidF(units)
    SigmoidF(units_minus)

    units.Mul(units, units_minus.T())
}

var Sigmoid Activation = Activation{Function: SigmoidF, Derivative: SigmoidDeriv}



func Linear(units *mat.VecDense){}


type Cost struct{
    Function func (*mat.Dense, *mat.Dense) (float64)
    Derivative func (*mat.Dense, *mat.Dense) (*mat.Dense)
}

func MSE(y_hat, y *mat.Dense) float64{
    var mses *mat.Dense
    mse := 0.0

    mses.Sub(y, y_hat)
    mses.Mul(mses, mses.T())

    n_examples, _ := mses.Dims()

    for i := 0; i < n_examples; i++ {
        mse += mses.At(i, 0)
    }
    mse /= float64(n_examples)

    return mse
}

func MSE_Deriv(y_hat, y *mat.Dense) *mat.VecDense{
    n_examples, _ := y.Dims()

    var mses *mat.Dense

    mses.Sub(y, y_hat)
    mses.Scale(2, mses)

    derivs := mat.NewVecDense(n_examples, nil)
    for e_i_mse := 0; e_i_mse < n_examples; e_i_mse++{
        sum_row := 0.0
        for e_i_mse_j := 0; e_i_mse_j < n_examples; e_i_mse_j++ {
            sum_row += mses.At(e_i_mse, e_i_mse_j)
        }
        sum_row /= float64(n_examples)

        derivs.SetVec(e_i_mse, sum_row)
    }

    return derivs
}

func GetMiniBatch(X_train, Y_train *mat.Dense, perm []int, mini_batch_i, mini_batch_size int) (*mat.Dense, *mat.Dense){
    _, n_features := X_train.Dims()
    _, n_classes := Y_train.Dims()

    mini_batch_X := mat.NewDense(mini_batch_size, n_features, nil)
    mini_batch_Y := mat.NewDense(mini_batch_size, n_classes, nil)

    for r := 0; r < mini_batch_size; r++{
        mini_batch_X.SetRow(r, X_train.RawRowView(perm[r + mini_batch_i]))
        mini_batch_Y.SetRow(r, Y_train.RawRowView(perm[r + mini_batch_i]))
    }

    return mini_batch_X, mini_batch_Y
}

