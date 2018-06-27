
package MLP

import (
    "gonum.org/v1/gonum/mat"
    . "dl/MLP/utils"
    "math/rand"
)




type MLP struct{
    N_Hiddens int
    Hiddens []*Dense
    In *mat.VecDense
    Out *Dense
    L Cost
}

func NewMLP(hidden_layers []Dense, input_size int, out_l *Dense) (*MLP){
    nn := &MLP{N_Hiddens: 0}
    nn.In = mat.NewVecDense(input_size, nil)

    for lj := 0; lj < len(hidden_layers); lj++{
        nn.addHiddenLayer(&hidden_layers[lj])
    }

    // for _,v := range(nn.Hiddens){
    //     MatPrint(v.Weights)
    // }

    nn.Out = out_l
    nn.Out.Init(hidden_layers[nn.N_Hiddens - 1].N_Units)

    return nn
}

func(nn *MLP) SetInput(in []float64) {
    in_one := make([]float64, len(in) + 1)
    in_one[0] = 1
    for i,_ := range(in){
        in_one[i] = in[i]
    }

    nn.In = mat.NewVecDense(len(in) + 1, in_one)
}

func(nn *MLP) addHiddenLayer(ly *Dense){
    nn.Hiddens = append(nn.Hiddens, ly)

    if nn.N_Hiddens > 1{
        ly.Init(nn.Hiddens[nn.N_Hiddens - 1].N_Units)
    }else{
        ly.Init(nn.In.Len())
    }

    nn.N_Hiddens++
}

func(nn *MLP) FeedForward() *mat.VecDense{
    nn.Hiddens[0].Units.MulVec(nn.Hiddens[0].Weights, nn.In)
    nn.Hiddens[0].Sigma(nn.Hiddens[0].Units)

    for lj := 1; lj < nn.N_Hiddens; lj++{
        nn.Hiddens[lj].Units.MulVec(nn.Hiddens[lj].Weights, nn.Hiddens[lj - 1].Units)
        nn.Hiddens[lj].Sigma(nn.Hiddens[lj].Units)
        //MatPrint(nn.Hiddens[lj].Units)
    }

    nn.Out.Units.MulVec(nn.Out.Weights, nn.Hiddens[nn.N_Hiddens - 1].Units)
    nn.Out.Sigma(nn.Out.Units)

    return nn.Out.Units
}


func(nn *MLP) FeedForwardBatch(batch *mat.Dense) []*mat.Dense{

    r_out, _  := batch.Dims()

    outputs := []*mat.Dense{}
    batch.T()

    outputs[0].Mul(nn.Hiddens[0].Weights, batch)

    for r := 0; r < r_out; r++{
        row := outputs[0].RowView(r)
        nn.Hiddens[0].Sigma(row)
        outputs[0].setRow(r, row)
    }

    for lj := 1; lj < nn.N_Hiddens; lj++{

        nn.Hiddens[lj].Units.MulVec(nn.Hiddens[lj].Weights, nn.Hiddens[lj - 1].Units)
        nn.Hiddens[lj].Sigma(nn.Hiddens[lj].Units)
        //MatPrint(nn.Hiddens[lj].Units)
    }

    nn.Out.Units.MulVec(nn.Out.Weights, nn.Hiddens[nn.N_Hiddens - 1].Units)
    nn.Out.Sigma(nn.Out.Units)

    return nn.Out.Units
}


func (nn *MLP) BackWardFeed(){

}


func (nn *MLP) fit(training_set *mat.Dense, epochs, mini_batch_size int, lr float64){
    population_size, n_features = training_set.Dims()

    for epoch := range(epochs){
        perm = rand.Perm(population_size)

        for mini_batch_i := 0; mini_batch_i < population_size;{
            mini_batch := GetMiniBatch(training_set, mini_batch_i, mini_batch_size)

            mini_batch_i = min(mini_batch_i + mini_batch_size, population_size)
        }
    }
}
