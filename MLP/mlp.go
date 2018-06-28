
package MLP

import (
    "gonum.org/v1/gonum/mat"
    . "goDeep/MLP/utils"
    "math/rand"
    "fmt"
)




type MLP struct{
    N_Hiddens, MiniBatchSize, Input_len int
    Hiddens []*Dense
    Out *Dense
    J Cost
}

func NewMLP(hidden_layers []Dense, input_size int, out_l *Dense) (*MLP){
    nn := &MLP{N_Hiddens: 0}
    nn.Input_len = input_size

    for lj := 0; lj < len(hidden_layers); lj++{
        nn.addHiddenLayer(&hidden_layers[lj])
    }

    // for _,v := range(nn.Hiddens){
    //     MatPrint(v.Weights)
    // }

    nn.Out = out_l
    nn.Out.Init(hidden_layers[nn.N_Hiddens - 1].N_Units, nn.MiniBatchSize)

    return nn
}

func(nn *MLP) addHiddenLayer(ly *Dense){
    nn.Hiddens = append(nn.Hiddens, ly)

    if nn.N_Hiddens > 1{
        ly.Init(nn.Hiddens[nn.N_Hiddens - 1].N_Units, nn.MiniBatchSize)
    }else{
        ly.Init(nn.Input_len, nn.MiniBatchSize)
    }

    nn.N_Hiddens++
}

/*func(nn *MLP) FeedForward() *mat.VecDense{
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
*/

func(nn *MLP) FeedForward(mini_batch *mat.Dense) *mat.Dense{

    nn.Hiddens[0].Units.Mul(nn.Hiddens[0].Weights, mini_batch.T())
    nn.Hiddens[0].G.Function(nn.Hiddens[0].Units)

    for lj := 1; lj < nn.N_Hiddens; lj++{
        nn.Hiddens[lj].Units.Mul(nn.Hiddens[lj].Weights, nn.Hiddens[lj - 1].Units.T())
        nn.Hiddens[lj].G.Function(nn.Hiddens[lj].Units)
        //MatPrint(nn.Hiddens[lj].Units)
    }

    nn.Out.Units.Mul(nn.Out.Weights, nn.Hiddens[nn.N_Hiddens - 1].Units.T())
    nn.Out.G.Function(nn.Out.Units)

    return nn.Out.Units
}


func (nn *MLP) BackwardFeed(y_hat *mat.Dense) float64{
    loss := nn.J.Function(nn.Out.Units, y_hat)

    return loss
}

func (nn *MLP) SGD(lr float64){

}

func (nn *MLP) Fit(X_train, Y_train *mat.Dense, epochs, mini_batch_size int, lr float64){
    population_size, _ := X_train.Dims()
    loss_train := 0.0

    for epoch := 0; epoch < epochs; epoch++ {
        perm := rand.Perm(population_size)
        for mini_batch_i := 0; mini_batch_i < population_size;{
            mini_batch_X, mini_batch_Y := GetMiniBatch(X_train, Y_train, perm, mini_batch_i, mini_batch_size)
            //compute nn output layer
            nn.FeedForward(mini_batch_X)
            //compute cost function & its derivative
            loss_train = nn.BackwardFeed(mini_batch_Y)

            //update weights
            nn.SGD(lr)
            mini_batch_i = Min(mini_batch_i + mini_batch_size, population_size)
        }

        fmt.Printf("End of epoch %v/%v, loss train = %v", epoch, epochs, loss_train)
    }

}
