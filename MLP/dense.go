package MLP
import (
    "gonum.org/v1/gonum/mat"
    . "dl/MLP/utils"
    "math/rand"
)
type Dense struct{
    N_Units int
    Units *mat.VecDense
    Weights *mat.Dense //biases are encoded in the weights, thus we make sur we 1-feature to the inputs
    Sigma Activation
}

func (l *Dense) InitWeights(){
    r, c := l.Weights.Dims()
    for ri := 0; ri < r; ri++{
        for ci := 0; ci < c; ci++{
            l.Weights.Set(ri, ci, rand.NormFloat64())
        }
    }
}

func(ly *Dense) Init(size_in int) {
    us := mat.NewVecDense(ly.N_Units, nil)
    w := mat.NewDense(ly.N_Units, size_in, nil)

    ly.Units = us
    ly.Weights = w
    ly.InitWeights()
}
