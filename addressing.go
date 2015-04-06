package ntm

import (
	"log"
	"math"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
)

type similarityCircuit struct {
	U   []Unit
	V   []Unit
	Top Unit

	UV    float64
	Unorm float64
	Vnorm float64
}

func newSimilarityCircuit(u, v []Unit) *similarityCircuit {
	s := similarityCircuit{
		U: u,
		V: v,
	}
	for i := 0; i < len(u); i++ {
		s.UV += u[i].Val * v[i].Val
		s.Unorm += u[i].Val * u[i].Val
		s.Vnorm += v[i].Val * v[i].Val
	}
	s.Unorm = math.Sqrt(s.Unorm)
	s.Vnorm = math.Sqrt(s.Vnorm)
	s.Top.Val = s.UV / (s.Unorm * s.Vnorm)
	if math.IsNaN(s.Top.Val) {
		log.Printf("u: %+v, v: %+v", u, v)
		panic("")
	}
	return &s
}

func (s *similarityCircuit) Backward() {
	uvuu := s.UV / (s.Unorm * s.Unorm)
	uvvv := s.UV / (s.Vnorm * s.Vnorm)
	uvg := s.Top.Grad / (s.Unorm * s.Vnorm)
	for i, u := range s.U {
		v := s.V[i].Val
		s.U[i].Grad += (v - u.Val*uvuu) * uvg
		s.V[i].Grad += (u.Val - v*uvvv) * uvg
	}
}

type betaSimilarity struct {
	Beta *Unit // Beta is assumed to be in the range (-Inf, Inf)
	S    *similarityCircuit
	Top  Unit

	b float64
}

func newBetaSimilarity(beta *Unit, s *similarityCircuit) *betaSimilarity {
	bs := betaSimilarity{
		Beta: beta,
		S:    s,
		b:    math.Exp(beta.Val),
	}
	bs.Top.Val = bs.b * s.Top.Val
	return &bs
}

func (bs *betaSimilarity) Backward() {
	bs.Beta.Grad += bs.S.Top.Val * bs.b * bs.Top.Grad
	bs.S.Top.Grad += bs.b * bs.Top.Grad
}

type contentAddressing struct {
	Units []*betaSimilarity
	Top   []Unit
}

func newContentAddressing(units []*betaSimilarity) *contentAddressing {
	s := contentAddressing{
		Units: units,
		Top:   make([]Unit, len(units)),
	}
	// Increase numerical stability by subtracting all weights by their max,
	// before computing math.Exp().
	var max float64 = -math.MaxFloat64
	for _, u := range s.Units {
		max = math.Max(max, u.Top.Val)
	}
	var sum float64 = 0
	for i, u := range s.Units {
		w := math.Exp(u.Top.Val - max)
		s.Top[i].Val = w
		sum += w
	}
	for i, top := range s.Top {
		s.Top[i].Val = top.Val / sum
	}
	return &s
}

func (s *contentAddressing) Backward() {
	var gv float64 = 0
	for _, top := range s.Top {
		gv += top.Grad * top.Val
	}
	for i, top := range s.Top {
		s.Units[i].Top.Grad += (top.Grad - gv) * top.Val
	}
}

type gatedWeighting struct {
	G    *Unit
	WC   *contentAddressing
	Wtm1 *refocus // the weights at time t-1
	Top  []Unit
}

func newGatedWeighting(g *Unit, wc *contentAddressing, wtm1 *refocus) *gatedWeighting {
	wg := gatedWeighting{
		G:    g,
		WC:   wc,
		Wtm1: wtm1,
		Top:  make([]Unit, len(wc.Top)),
	}
	gt := Sigmoid(g.Val)
	for i := 0; i < len(wg.Top); i++ {
		wg.Top[i].Val = gt*wc.Top[i].Val + (1-gt)*wtm1.TopVal[i]
	}
	return &wg
}

func (wg *gatedWeighting) Backward() {
	gt := Sigmoid(wg.G.Val)

	var grad float64 = 0
	for i := 0; i < len(wg.Top); i++ {
		grad += (wg.WC.Top[i].Val - wg.Wtm1.TopVal[i]) * wg.Top[i].Grad
	}
	wg.G.Grad += grad * gt * (1 - gt)

	for i := 0; i < len(wg.WC.Top); i++ {
		wg.WC.Top[i].Grad += gt * wg.Top[i].Grad
	}

	for i := 0; i < len(wg.Wtm1.TopGrad); i++ {
		wg.Wtm1.TopGrad[i] += (1 - gt) * wg.Top[i].Grad
	}
}

type shiftedWeighting struct {
	S   *Unit
	Z   float64
	WG  *gatedWeighting
	Top []Unit
}

func newShiftedWeighting(s *Unit, wg *gatedWeighting) *shiftedWeighting {
	sw := shiftedWeighting{
		S:   s,
		WG:  wg,
		Top: make([]Unit, len(wg.Top)),
	}

	n := len(sw.WG.Top)
	//sw.Z = math.Mod(s.Val, float64(n))
	//if sw.Z < 0 {
	//	sw.Z += float64(n)
	//}

	//sw.Z = float64(n) * Sigmoid(s.Val)
	shift := (2*Sigmoid(s.Val) - 1) // * maxShift
	sw.Z = math.Mod(shift+float64(n), float64(n))

	simj := 1 - (sw.Z - math.Floor(sw.Z))
	for i := 0; i < len(sw.Top); i++ {
		imj := (i + int(sw.Z)) % n
		sw.Top[i].Val = sw.WG.Top[imj].Val*simj + sw.WG.Top[(imj+1)%n].Val*(1-simj)
		if math.IsNaN(sw.Top[i].Val) || sw.Top[i].Val < 0 {
			log.Printf("imj: %d, wg: %f, simj: %f, wg+1: %f", imj, sw.WG.Top[imj].Val, simj, sw.WG.Top[(imj+1)%n].Val)
			panic("")
		}
	}
	return &sw
}

func (sw *shiftedWeighting) Backward() {
	var grad float64 = 0
	n := len(sw.WG.Top)
	for i := 0; i < len(sw.Top); i++ {
		imj := (i + int(sw.Z)) % n
		grad += (-sw.WG.Top[imj].Val + sw.WG.Top[(imj+1)%n].Val) * sw.Top[i].Grad
	}
	sig := Sigmoid(sw.S.Val)
	grad = grad * 2 * sig * (1 - sig)
	// grad = grad * sw.Z * (1 - sw.Z/float64(n))
	sw.S.Grad += grad

	simj := 1 - (sw.Z - math.Floor(sw.Z))
	for i := 0; i < len(sw.WG.Top); i++ {
		j := (i - int(sw.Z) + n) % n
		sw.WG.Top[i].Grad += sw.Top[j].Grad*simj + sw.Top[(j-1+n)%n].Grad*(1-simj)
	}
}

type refocus struct {
	Gamma *Unit
	SW    *shiftedWeighting

	TopVal  []float64
	TopGrad []float64

	g float64
}

func newRefocus(gamma *Unit, sw *shiftedWeighting) *refocus {
	rf := refocus{
		Gamma:   gamma,
		SW:      sw,
		TopVal:  make([]float64, len(sw.Top)),
		TopGrad: make([]float64, len(sw.Top)),
		g:       math.Log(math.Exp(gamma.Val)+1) + 1,
	}
	var sum float64 = 0
	for i := 0; i < len(rf.TopVal); i++ {
		rf.TopVal[i] = math.Pow(sw.Top[i].Val, rf.g)
		sum += rf.TopVal[i]
	}
	for i := 0; i < len(rf.TopVal); i++ {
		rf.TopVal[i] = rf.TopVal[i] / sum
	}
	return &rf
}

func (rf *refocus) backwardSW() {
	var topGV float64 = 0
	for i, topV := range rf.TopVal {
		topGV += rf.TopGrad[i] * topV
	}
	for i, sw := range rf.SW.Top {
		if sw.Val < machineEpsilon {
			continue
		}
		rf.SW.Top[i].Grad += (rf.TopGrad[i] - topGV) * rf.g / sw.Val * rf.TopVal[i]
	}
}

func (rf *refocus) backwardGamma() {
	lns := make([]float64, len(rf.SW.Top))
	var lnexp float64 = 0
	var s float64 = 0
	for i, sw := range rf.SW.Top {
		if sw.Val < machineEpsilon {
			continue
		}
		lns[i] = math.Log(sw.Val)
		pow := math.Pow(sw.Val, rf.g)
		lnexp += lns[i] * pow
		s += pow
	}
	lnexps := lnexp / s
	var grad float64 = 0
	for i, topV := range rf.TopVal {
		if rf.SW.Top[i].Val < machineEpsilon {
			continue
		}
		grad += rf.TopGrad[i] * (topV * (lns[i] - lnexps))
	}
	grad = grad / (1 + math.Exp(-rf.Gamma.Val))
	rf.Gamma.Grad += grad
}

func (rf *refocus) Backward() {
	rf.backwardSW()
	rf.backwardGamma()
}

type memRead struct {
	W      *refocus
	Memory *writtenMemory
	Top    []Unit
}

func newMemRead(w *refocus, memory *writtenMemory) *memRead {
	m := len(memory.TopVal) / memory.N
	r := memRead{
		W:      w,
		Memory: memory,
		Top:    make([]Unit, m),
	}

	for i := 0; i < len(r.Top); i++ {
		var v float64 = 0
		for j := 0; j < len(w.TopVal); j++ {
			v += w.TopVal[j] * memory.TopVal[j*m+i]
		}
		r.Top[i].Val = v
	}
	return &r
}

func (r *memRead) Backward() {
	n := r.Memory.N
	m := len(r.Memory.TopVal) / n

	for i := 0; i < n; i++ {
		var grad float64 = 0
		for j := 0; j < m; j++ {
			grad += r.Top[j].Grad * r.Memory.TopVal[i*m+j]
		}
		r.W.TopGrad[i] += grad
	}

	for i := 0; i < n; i++ {
		w := r.W.TopVal[i]
		for j := 0; j < m; j++ {
			r.Memory.TopGrad[i*m+j] += r.Top[j].Grad * w
		}
	}
}

type writtenMemory struct {
	Ws    []*refocus
	Heads []*Head        // We actually need only the erase and add vectors.
	Mtm1  *writtenMemory // memory at time t-1

	N       int // memoryN
	TopVal  []float64
	TopGrad []float64

	erase    [][]float64
	add      [][]float64
	erasures []float64
}

func newWrittenMemory(ws []*refocus, heads []*Head, mtm1 *writtenMemory) *writtenMemory {
	n := mtm1.N
	m := len(mtm1.TopVal) / n
	wm := writtenMemory{
		Ws:    ws,
		Heads: heads,
		Mtm1:  mtm1,

		N:       mtm1.N,
		TopVal:  make([]float64, len(mtm1.TopVal)),
		TopGrad: make([]float64, len(mtm1.TopVal)),

		erase:    MakeTensor2(len(heads), m),
		add:      MakeTensor2(len(heads), m),
		erasures: make([]float64, len(mtm1.TopVal)),
	}
	for i, h := range wm.Heads {
		erase := wm.erase[i]
		add := wm.add[i]
		eraseVec := h.EraseVector()
		addVec := h.AddVector()
		for j, e := range eraseVec {
			erase[j] = Sigmoid(e.Val)
			add[j] = Sigmoid(addVec[j].Val)
		}
	}

	copy(wm.erasures, mtm1.TopVal)
	we := make([]float64, n*m)
	weG := blas64.General{Rows: n, Cols: m, Stride: m, Data: we}
	for k, ws := range wm.Ws {
		weights := blas64.Vector{Inc: 1, Data: ws.TopVal}
		erase := blas64.Vector{Inc: 1, Data: wm.erase[k]}
		for i := range we {
			we[i] = 1
		}
		blas64.Ger(-1, weights, erase, weG)
		Mul(wm.erasures, we)
	}

	copy(wm.TopVal, wm.erasures)
	topG := blas64.General{Rows: n, Cols: m, Stride: m, Data: wm.TopVal}
	for k, ws := range wm.Ws {
		weights := blas64.Vector{Inc: 1, Data: ws.TopVal}
		add := blas64.Vector{Inc: 1, Data: wm.add[k]}
		blas64.Ger(1, weights, add, topG)
	}

	return &wm
}

func (wm *writtenMemory) div1MWE(out []float64) {
	m := len(wm.TopVal) / wm.N
	for i, e := range wm.erasures {
		mwe := 1 - out[i]
		if math.Abs(mwe) > 1e-6 {
			out[i] = e / mwe
		} else {
			j := i / m
			k := i % m
			mtilt := wm.Mtm1.TopVal[j*m+k]
			for q, ws := range wm.Ws {
				if q == i {
					continue
				}
				mtilt = mtilt * (1 - ws.TopVal[j]*wm.erase[q][k])
			}
			out[i] = mtilt
		}
	}
}

func (wm *writtenMemory) backwardWErase() {
	n := wm.N
	m := len(wm.TopVal) / n

	mgrad := make([]float64, n*m)
	mGradG := blas64.General{Rows: n, Cols: m, Stride: m, Data: mgrad}
	hEraseGrad := blas64.Vector{Inc: 1, Data: make([]float64, len(wm.Heads[0].EraseVector()))}
	for i, weights := range wm.Ws {
		erase := wm.erase[i]
		add := wm.add[i]
		eraseV := blas64.Vector{Inc: 1, Data: erase}
		addV := blas64.Vector{Inc: 1, Data: add}
		weightsVal := blas64.Vector{Inc: 1, Data: weights.TopVal}

		for j := range mgrad {
			mgrad[j] = 0
		}
		blas64.Ger(1, weightsVal, eraseV, mGradG)
		wm.div1MWE(mgrad)
		Mul(mgrad, wm.TopGrad)

		weightsV := blas64.Vector{Inc: 1, Data: weights.TopGrad}
		blas64.Gemv(blas.NoTrans, -1, mGradG, eraseV, 1, weightsV)
		blas64.Gemv(blas.NoTrans, 1, blas64.General{Rows: n, Cols: m, Stride: m, Data: wm.TopGrad}, addV, 1, weightsV)

		hErase := wm.Heads[i].EraseVector()
		for j := range hEraseGrad.Data {
			hEraseGrad.Data[j] = 0
		}
		blas64.Gemv(blas.Trans, -1, mGradG, weightsVal, 1, hEraseGrad)
		for j, e := range erase {
			hErase[j].Grad += hEraseGrad.Data[j] * e * (1 - e)
		}
	}
}

func (wm *writtenMemory) backwardAdd() {
	n := wm.N
	m := len(wm.TopVal) / n

	var grad float64
	for k, h := range wm.Heads {
		add := wm.add[k]
		ws := wm.Ws[k]
		hAdd := h.AddVector()
		for i := range hAdd {
			grad = 0
			for j := 0; j < n; j++ {
				grad += wm.TopGrad[j*m+i] * ws.TopVal[j]
			}
			a := add[i]
			hAdd[i].Grad += grad * a * (1 - a)
		}
	}
}

func (wm *writtenMemory) backwardMtm1() {
	n := wm.N
	m := len(wm.TopVal) / n

	var grad float64
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			grad = 1
			for q, ws := range wm.Ws {
				grad = grad * (1 - ws.TopVal[i]*wm.erase[q][j])
			}
			wm.Mtm1.TopGrad[i*m+j] += grad * wm.TopGrad[i*m+j]
		}
	}
}

func (wm *writtenMemory) Backward() {
	wm.backwardWErase()
	wm.backwardAdd()
	wm.backwardMtm1()
}

type memOp struct {
	W  []*refocus
	R  []*memRead
	WM *writtenMemory
}

func newMemOp(heads []*Head, mtm1 *writtenMemory) *memOp {
	circuit := memOp{
		R: make([]*memRead, len(heads)),
	}
	circuit.W = make([]*refocus, len(heads))
	for wi, h := range heads {
		ss := make([]*betaSimilarity, mtm1.N)
		for i := 0; i < mtm1.N; i++ {
			m := len(mtm1.TopVal) / mtm1.N
			units := make([]Unit, m)
			for j := range units {
				units[j].Val = mtm1.TopVal[i*m+j]
			}

			s := newSimilarityCircuit(h.K(), units)
			ss[i] = newBetaSimilarity(h.Beta(), s)
		}
		wc := newContentAddressing(ss)
		wg := newGatedWeighting(h.G(), wc, h.Wtm1)
		ws := newShiftedWeighting(h.S(), wg)
		circuit.W[wi] = newRefocus(h.Gamma(), ws)
		circuit.R[wi] = newMemRead(circuit.W[wi], mtm1)
	}

	circuit.WM = newWrittenMemory(circuit.W, heads, mtm1)
	return &circuit
}

func (c *memOp) Backward() {
	for _, r := range c.R {
		r.Backward()
	}
	c.WM.Backward()

	for _, rf := range c.WM.Ws {
		rf.Backward()
		rf.SW.Backward()
		rf.SW.WG.Backward()
		rf.SW.WG.WC.Backward()
		for i, bs := range rf.SW.WG.WC.Units {
			bs.Backward()
			bs.S.Backward()

			m := len(c.WM.TopVal) / c.WM.N
			for j := range bs.S.V {
				c.WM.Mtm1.TopGrad[i*m+j] += bs.S.V[j].Grad
			}
		}
	}
}

//func (c *memOp) ReadVals() [][]float64 {
//	res := MakeTensor2(len(c.R), len(c.R[0].Top))
//	for i := 0; i < len(res); i++ {
//		for j := 0; j < len(res[i]); j++ {
//			res[i][j] = c.R[i].Top[j].Val
//		}
//	}
//	return res
//}
//
//func (c *memOp) WrittenMemoryVals() [][]float64 {
//	res := MakeTensor2(len(c.WM.Top), len(c.WM.Top[0]))
//	for i := 0; i < len(res); i++ {
//		for j := 0; j < len(res[i]); j++ {
//			res[i][j] = c.WM.Top[i][j].Val
//		}
//	}
//	return res
//}
