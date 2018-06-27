package MLP
import (
    "gonum.org/v1/gonum/mat"
    . "goDeep/MLP/utils"
    "math/rand"
)
type Dense struct{
    N_Units int
    Units, Weights, Delta *mat.Dense
    G Activation
}

func (l *Dense) InitWeights(){
    r, c := l.Weights.Dims()
    for ri := 0; ri < r; ri++{
        for ci := 0; ci < c; ci++{
            l.Weights.Set(ri, ci, rand.NormFloat64())
        }
    }
}

func(ly *Dense) Init(size_in, mini_batch_size int) {
    ly.Units = mat.NewDense(mini_batch_size, ly.N_Units, nil)
    ly.Weights = mat.NewDense(ly.N_Units, size_in, nil)
    ly.Delta = mat.NewDense(ly.N_Units, size_in, nil)

    ly.InitWeights()

}
