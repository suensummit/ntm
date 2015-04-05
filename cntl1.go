package ntm

import (
	"fmt"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
)

type controller1 struct {
	weightsVal  []float64
	weightsGrad []float64

	Reads     []*memRead
	X         []float64
	ReadsXVal blas64.Vector

	H1Val  []float64
	H1Grad []float64

	y     []Unit
	heads []*Head

	numHeads int
	memoryM  int
	memoryN  int
	xSize    int
	h1Size   int
	ySize    int
}

func (c *controller1) wh1Cols() int {
	return c.numHeads*c.memoryM + c.xSize + 1
}

func (c *controller1) wyRows() int {
	return c.ySize + c.numHeads*headUnitsLen(c.memoryM)
}

func (c *controller1) wyOffset() int {
	return c.h1Size * c.wh1Cols()
}

func (c *controller1) wtm1Offset() int {
	return c.wyOffset() + c.wyRows()*(c.h1Size+1)
}

func (c *controller1) mtm1Offset() int {
	return c.wtm1Offset() + c.numHeads*c.memoryN
}

func (c *controller1) NumWeights() int {
	return c.mtm1Offset() + c.memoryN*c.memoryM
}

func (c *controller1) wh1(w []float64) blas64.General {
	m := blas64.General{
		Rows: c.h1Size,
		Cols: c.wh1Cols(),
	}
	m.Stride = m.Cols
	m.Data = w[0:c.wyOffset()]
	return m
}

func (c *controller1) wh1Val() blas64.General {
	return c.wh1(c.weightsVal)
}

func (c *controller1) wh1Grad() blas64.General {
	return c.wh1(c.weightsGrad)
}

func (c *controller1) wy(w []float64) blas64.General {
	m := blas64.General{
		Rows: c.wyRows(),
		Cols: c.h1Size + 1,
	}
	m.Stride = m.Cols
	m.Data = w[c.wyOffset():c.wtm1Offset()]
	return m
}

func (c *controller1) wyVal() blas64.General {
	return c.wy(c.weightsVal)
}

func (c *controller1) wyGrad() blas64.General {
	return c.wy(c.weightsGrad)
}

func (c *controller1) Wtm1BiasVal() []float64 {
	return c.weightsVal[c.wtm1Offset():c.mtm1Offset()]
}

func (c *controller1) Wtm1BiasGrad() []float64 {
	return c.weightsGrad[c.wtm1Offset():c.mtm1Offset()]
}

func (c *controller1) Mtm1BiasVal() []float64 {
	return c.weightsVal[c.mtm1Offset():]
}

func (c *controller1) Mtm1BiasGrad() []float64 {
	return c.weightsGrad[c.mtm1Offset():]
}

// NewEmptyController1 returns a new controller1 which is a single layer feedforward network.
// The returned controller1 is empty in that all its network weights are initialized as 0.
func NewEmptyController1(xSize, ySize, h1Size, numHeads, n, m int) *controller1 {
	c := controller1{
		numHeads: numHeads,
		memoryM:  m,
		memoryN:  n,
		xSize:    xSize,
		h1Size:   h1Size,
		ySize:    ySize,
	}
	c.weightsVal = make([]float64, c.NumWeights())
	c.weightsGrad = make([]float64, c.NumWeights())
	return &c
}

func (c *controller1) Heads() []*Head {
	return c.heads
}

func (c *controller1) Y() []Unit {
	return c.y
}

func (old *controller1) Forward(reads []*memRead, x []float64) Controller {
	c := controller1{
		weightsVal:  old.weightsVal,
		weightsGrad: old.weightsGrad,
		Reads:       reads,
		X:           x,
		H1Val:       make([]float64, old.h1Size+1),
		H1Grad:      make([]float64, old.h1Size+1),
		y:           make([]Unit, old.ySize),
		heads:       make([]*Head, len(reads)),

		numHeads: old.numHeads,
		memoryM:  old.memoryM,
		memoryN:  old.memoryN,
		xSize:    old.xSize,
		h1Size:   old.h1Size,
		ySize:    old.ySize,
	}

	ud := make([]float64, c.wh1Cols())
	for i, read := range reads {
		for j, r := range read.Top {
			ud[i*c.memoryM+j] = r.Val
		}
	}
	copy(ud[c.numHeads*c.memoryM:], c.X)
	ud[c.numHeads*c.memoryM+c.xSize] = 1
	c.ReadsXVal = blas64.Vector{Inc: 1, Data: ud}

	h1 := blas64.Vector{Inc: 1, Data: c.H1Val[0:c.h1Size]}
	blas64.Gemv(blas.NoTrans, 1, c.wh1Val(), c.ReadsXVal, 1, h1)
	for i, h := range c.H1Val[0:c.h1Size] {
		c.H1Val[i] = Sigmoid(h)
	}

	c.H1Val[c.h1Size] = 1
	h1 = blas64.Vector{Inc: 1, Data: c.H1Val}
	out := blas64.Vector{Inc: 1, Data: make([]float64, c.wyRows())}
	blas64.Gemv(blas.NoTrans, 1, c.wyVal(), h1, 1, out)

	for i, v := range out.Data[0:c.ySize] {
		c.y[i].Val = v
	}
	hul := headUnitsLen(c.memoryM)
	for i := range c.heads {
		head := NewHead(c.memoryM)
		c.heads[i] = head
		start := c.ySize + i*hul
		for j, v := range out.Data[start : start+hul] {
			head.units[j].Val = v
		}
	}

	return &c
}

func (c *controller1) Backward() {
	out := blas64.Vector{Inc: 1, Data: make([]float64, c.wyRows())}
	for i, v := range c.y {
		out.Data[i] = v.Grad
	}
	hul := headUnitsLen(c.memoryM)
	for i, head := range c.heads {
		start := c.ySize + i*hul
		for j, v := range head.units {
			out.Data[start+j] = v.Grad
		}
	}

	h1Val := blas64.Vector{Inc: 1, Data: c.H1Val}
	h1Grad := blas64.Vector{Inc: 1, Data: c.H1Grad}
	blas64.Gemv(blas.Trans, 1, c.wyVal(), out, 1, h1Grad)
	blas64.Ger(1, out, h1Val, c.wyGrad())

	h1Val = blas64.Vector{Inc: 1, Data: c.H1Val[0:c.h1Size]}
	h1Grad = blas64.Vector{Inc: 1, Data: c.H1Grad[0:c.h1Size]}
	for i, v := range h1Val.Data {
		h1Grad.Data[i] *= v * (1 - v)
	}

	u := blas64.Vector{Inc: 1, Data: make([]float64, c.wh1Cols())}
	blas64.Gemv(blas.Trans, 1, c.wh1Val(), h1Grad, 1, u)
	blas64.Ger(1, h1Grad, c.ReadsXVal, c.wh1Grad())

	for i, read := range c.Reads {
		for j, g := range u.Data[i*c.memoryM : (i+1)*c.memoryM] {
			read.Top[j].Grad = g
		}
	}
}

func (c *controller1) WeightsVal() []float64 {
	return c.weightsVal
}

func (c *controller1) WeightsGrad() []float64 {
	return c.weightsGrad
}

func (c *controller1) WeightsDesc(i int) string {
	if i < c.wyOffset() {
		return fmt.Sprintf("wh1[%d][%d]", i/c.wh1Cols(), i%c.wh1Cols())
	}
	if i < c.wtm1Offset() {
		j := i - c.wyOffset()
		cols := c.h1Size + 1
		return fmt.Sprintf("wy[%d][%d]", j/cols, i%cols)
	}
	if i < c.mtm1Offset() {
		j := i - c.wtm1Offset()
		return fmt.Sprintf("wtm1[%d][%d]", j/c.memoryN, j%c.memoryN)
	}
	j := i - c.mtm1Offset()
	return fmt.Sprintf("mtm1[%d][%d]", j/c.memoryM, j%c.memoryM)
}

func (c *controller1) NumHeads() int {
	return c.numHeads
}

func (c *controller1) MemoryN() int {
	return c.memoryN
}

func (c *controller1) MemoryM() int {
	return c.memoryM
}
