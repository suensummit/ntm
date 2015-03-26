/*
Package ntm implements the Neural Turing Machine architecture as described in A.Graves, G. Wayne, and I. Danihelka. arXiv preprint arXiv:1410.5401, 2014.

Using this package along its subpackages, the "copy", "repeatcopy" and "ngram" tasks mentioned in the paper were verified.
For each of these tasks, the successfully trained models are saved under the filenames "seedA_B",
where A is the number indicating the seed provided to rand.Seed in the training process, and B is the iteration number in which the trained weights converged.
*/
package ntm

import (
	"math"
)

// A Head is a read write head on a memory bank.
// It carriess every information that is required to operate on a memory bank according to the NTM architecture..
type Head struct {
	units []Unit
	Wtm1  *refocus // the weights at time t-1
	M     int      // size of a row in the memory
}

// NewHead creates a new memory head.
func NewHead(m int) *Head {
	h := Head{
		units: make([]Unit, 3*m+4),
		M:     m,
	}
	return &h
}

// EraseVector returns the erase vector of a memory head.
func (h *Head) EraseVector() []Unit {
	return h.units[0:h.M]
}

// AddVector returns the add vector of a memory head.
func (h *Head) AddVector() []Unit {
	return h.units[h.M : 2*h.M]
}

// K returns a head's key vector, which is the target data in the content addressing step.
func (h *Head) K() []Unit {
	return h.units[2*h.M : 3*h.M]
}

// Beta returns the key strength of a content addressing step.
func (h *Head) Beta() *Unit {
	return &h.units[3*h.M]
}

// G returns the degree in which we want to choose content-addressing over location-based-addressing.
func (h *Head) G() *Unit {
	return &h.units[3*h.M+1]
}

// S returns a value indicating how much the weightings are rotated in a location-based-addressing step.
func (h *Head) S() *Unit {
	return &h.units[3*h.M+2]
}

// Gamma returns the degree in which the addressing weights are sharpened.
func (h *Head) Gamma() *Unit {
	return &h.units[3*h.M+3]
}

// The Controller interface is implemented by NTM controller networks that wish to operate with memory banks in a NTM.
type Controller interface {
	// Heads returns the emitted memory heads.
	Heads() []*Head
	// Y returns the output of the Controller.
	Y() []Unit

	// Forward creates a new Controller which shares the same internal weights,
	// and performs a forward pass whose results can be retrived by Heads and Y.
	Forward(reads []*memRead, x []float64) Controller
	// Backward performs a backward pass,
	// assuming the gradients on Heads and Y are already set.
	Backward()

	// Wtm1BiasV returns the bias values for the memory heads at time t-1.
	Wtm1BiasV() [][]*betaSimilarity
	// Mtm1BiasV returns the bias values for the memory at time t-1.
	Mtm1BiasV() *writtenMemory

	// Weights loops through all internal weights of a controller.
	// For each weight, Weights calls the callback with a unique tag and a pointer to the weight.
	Weights(f func(*Unit))
	WeightsVerbose(f func(string, *Unit))

	NumWeights() int
	NumHeads() int
	MemoryN() int
	MemoryM() int
}

// A NTM is a neural turing machine as described in A.Graves, G. Wayne, and I. Danihelka. arXiv preprint arXiv:1410.5401, 2014.
type NTM struct {
	Controller Controller
	memOp      *memOp
}

// NewNTM creates a new NTM.
func NewNTM(old *NTM, x []float64) *NTM {
	m := NTM{
		Controller: old.Controller.Forward(old.memOp.R, x),
	}
	for i := 0; i < len(m.Controller.Heads()); i++ {
		m.Controller.Heads()[i].Wtm1 = old.memOp.W[i]
	}
	m.memOp = newMemOp(m.Controller.Heads(), old.memOp.WM)
	return &m
}

func (m *NTM) backward() {
	m.memOp.Backward()
	m.Controller.Backward()
}

// ForwardBackward computes a controller's prediction and gradients with respect to the given ground truth input and output values.
func ForwardBackward(c Controller, in [][]float64, out DensityModel) []*NTM {
	// Set the empty NTM's memory and head weights to their bias values.
	empty, reads, cas := MakeEmptyNTM(c)
	machines := make([]*NTM, len(in))

	// Backpropagation through time.
	machines[0] = NewNTM(empty, in[0])
	for t := 1; t < len(in); t++ {
		machines[t] = NewNTM(machines[t-1], in[t])
	}
	c.Weights(func(u *Unit) { u.Grad = 0 })
	for t := len(in) - 1; t >= 0; t-- {
		m := machines[t]
		out.Model(t, m.Controller.Y())
		m.backward()
	}

	// Compute gradients for the bias values of the initial memory and weights.
	for i := range reads {
		reads[i].Backward()
		for j := range reads[i].W.Top {
			cas[i].Top[j].Grad += reads[i].W.Top[j].Grad
		}
		cas[i].Backward()
	}

	return machines
}

// MakeEmptyNTM makes a NTM with its memory and head weights set to their bias values, based on the controller.
func MakeEmptyNTM(c Controller) (*NTM, []*memRead, []*contentAddressing) {
	wtm1s := make([]*refocus, c.NumHeads())
	reads := make([]*memRead, c.NumHeads())
	cas := make([]*contentAddressing, c.NumHeads())
	for i := range reads {
		cas[i] = newContentAddressing(c.Wtm1BiasV()[i])
		wtm1s[i] = &refocus{Top: make([]Unit, c.MemoryN())}
		for j := range wtm1s[i].Top {
			wtm1s[i].Top[j].Val = cas[i].Top[j].Val
		}
		reads[i] = newMemRead(wtm1s[i], c.Mtm1BiasV())
	}

	empty := &NTM{
		Controller: c,
		memOp:      &memOp{W: wtm1s, R: reads, WM: c.Mtm1BiasV()},
	}

	return empty, reads, cas
}

// Predictions returns the predictions of a NTM across time.
func Predictions(machines []*NTM) [][]float64 {
	pdts := make([][]float64, len(machines))
	for t := range pdts {
		y := machines[t].Controller.Y()
		pdts[t] = UnitVals(y)
	}
	return pdts
}

// HeadWeights returns the addressing weights of all memory heads across time.
// The top level elements represent each head.
// The second level elements represent every time instant.
func HeadWeights(machines []*NTM) [][][]float64 {
	hws := make([][][]float64, len(machines[0].memOp.W))
	for i := range hws {
		hws[i] = make([][]float64, len(machines))
		for t, m := range machines {
			hws[i][t] = make([]float64, len(m.memOp.W[i].Top))
			for j, w := range m.memOp.W[i].Top {
				hws[i][t][j] = w.Val
			}
		}
	}
	return hws
}

// SGDMomentum implements stochastic gradient descent with momentum.
type SGDMomentum struct {
	C     Controller
	PrevD []float64
}

func NewSGDMomentum(c Controller) *SGDMomentum {
	s := SGDMomentum{
		C:     c,
		PrevD: make([]float64, c.NumWeights()),
	}
	return &s
}

func (s *SGDMomentum) Train(x [][]float64, y DensityModel, alpha, mt float64) []*NTM {
	machines := ForwardBackward(s.C, x, y)
	i := 0
	s.C.Weights(func(w *Unit) {
		d := -alpha*w.Grad + mt*s.PrevD[i]
		w.Val += d
		s.PrevD[i] = d
		i++
	})
	return machines
}

// RMSProp implements the rmsprop algorithm. The detailed updating equations are given in
// Graves, Alex (2013). Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850.
type RMSProp struct {
	C Controller
	N []float64
	G []float64
	D []float64
}

func NewRMSProp(c Controller) *RMSProp {
	r := RMSProp{
		C: c,
		N: make([]float64, c.NumWeights()),
		G: make([]float64, c.NumWeights()),
		D: make([]float64, c.NumWeights()),
	}
	return &r
}

func (r *RMSProp) Train(x [][]float64, y DensityModel, a, b, c, d float64) []*NTM {
	machines := ForwardBackward(r.C, x, y)
	i := 0
	r.C.Weights(func(w *Unit) {
		rN := a*r.N[i] + (1-a)*w.Grad*w.Grad
		r.N[i] = rN

		rG := a*r.G[i] + (1-a)*w.Grad
		r.G[i] = rG

		rD := b*r.D[i] - c*w.Grad/math.Sqrt(rN-rG*rG+d)
		r.D[i] = rD

		w.Val += rD
		i++
	})
	return machines
}

// An DensityModel is a model of how the last layer of a network gets transformed into the final output.
type DensityModel interface {
	// Model sets the value and gradient of Units of the output layer.
	Model(t int, yH []Unit)
	// Loss is the loss definition of this model.
	Loss(output [][]float64) float64
}

// A LogisticModel models its outputs as logistic sigmoids.
type LogisticModel struct {
	// Y is the strength of the output unit at each time step.
	Y [][]float64
}

// Model sets the values and gradients of the output units.
func (m *LogisticModel) Model(t int, yHs []Unit) {
	ys := m.Y[t]
	for i, yh := range yHs {
		u := Unit{Val: Sigmoid(yh.Val)}
		u.Grad = u.Val - ys[i]
		yHs[i] = u
	}
}

// Loss returns the cross entropy loss.
func (m *LogisticModel) Loss(output [][]float64) float64 {
	var l float64 = 0
	for t, yh := range output {
		for i, _ := range yh {
			p := output[t][i]
			y := m.Y[t][i]
			l += y*math.Log(p) + (1-y)*math.Log(1-p)
		}
	}
	return -l
}

// A MultinomialModel models its outputs as following the multinomial distribution.
type MultinomialModel struct {
	// Y is the class of the output at each time step.
	Y []int
}

// Model sets the values and gradients of the output units.
func (m *MultinomialModel) Model(t int, yHs []Unit) {
	var sum float64 = 0
	for i, yh := range yHs {
		v := math.Exp(yh.Val)
		yHs[i].Val = v
		sum += v
	}

	k := m.Y[t]
	for i, yh := range yHs {
		u := Unit{Val: yh.Val / sum}
		u.Grad = u.Val - delta(i, k)
		yHs[i] = u
	}
}

func (m *MultinomialModel) Loss(output [][]float64) float64 {
	var l float64 = 0
	for t, yh := range output {
		l += math.Log(yh[m.Y[t]])
	}
	return -l
}
