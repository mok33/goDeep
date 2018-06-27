package utils

import (
    "gonum.org/v1/gonum/mat"
    "fmt"
    "math"
    )

func min(a, b int) int {
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

func Sigmoid(units *mat.Dense) {
    size_out, size_in := units.Dims()

    for r := 0; r < size_in; r++{
        for c := 0; c < size_out; c++{
            v := units.At(r, c)
            units.Set(r, c, 1 /(1 + math.Exp(-1 * v)))
        }
    }
}

func Linear(units *mat.VecDense){}


type Cost func(*mat.Dense, *mat.Dense) (float64)

func MSE(y_hat, y *mat.Dense) *mat.Dense{
    var mse *mat.Dense

    mse.Sub(y, y_hat)
    mse.Mul(mse, mse.T())
    return mse
}

func GetMiniBatch(population *mat.Dense, mini_batch_i, mini_batch_size int) *mat.Dense{
    _, c := population.Dims()

    mini_batch := mat.NewDense(mini_batch_size, c, nil)
    for r := 0; r < mini_batch_size; r++{
        mini_batch.SetRow(r, population.RawRowView(r + mini_batch_i))
    }

    return mini_batch
}

