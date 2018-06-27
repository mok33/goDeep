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


type Activation func(units *mat.VecDense)

func Sigmoid(units *mat.VecDense) {
    for i := 0; i < units.Len(); i++{
        v := units.At(i, 0)
        units.SetVec(i, 1 /(1 + math.Exp(-1 * v)))
    }
}

func Linear(units *mat.VecDense){}


type Cost func(*mat.VecDense, *mat.VecDense) (float64)

func MSE(y_hat *mat.VecDense, y *mat.VecDense) (float64){
    var mse *mat.VecDense
    mse.SubVec(y, y_hat)
    return mat.Dot(mse, mse)
}

func GetMiniBatch(population *mat.Dense, mini_batch_i, mini_batch_size int) *mat.Dense{
    _, c := population.Dims()

    mini_batch := mat.NewDense(mini_batch_size, c, nil)
    for r := 0; r < mini_batch_size; r++{
        mini_batch.SetRow(r, population.RawRowView(r + mini_batch_i))
    }

    return mini_batch
}
