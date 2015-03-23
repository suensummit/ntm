package poem

import (
	"encoding/json"
	"math/rand"
	"os"
)

type Dataset struct {
	Chars map[string]int
	Shis  [][][]int
}

type Generator struct {
	Dataset Dataset

	indices []int
	offset  int
}

func NewGenerator(filepath string) (*Generator, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	g := Generator{}
	if err := json.NewDecoder(f).Decode(&g.Dataset); err != nil {
		return nil, err
	}
	g.indices = make([]int, len(g.Dataset.Shis))
	g.resample()
	return &g, nil
}

func (g *Generator) GenSeq() (x, y [][]float64) {
	poem := g.Dataset.Shis[g.indices[g.offset]]
	g.offset += 1
	if g.offset == len(g.indices) {
		g.resample()
	}

	// Limit poem size to avoid memory issues.
	// For poems with lines over 200, we might need over 10GB.
	if len(poem) > 32 {
		poem = poem[0:32]
	}

	inputSize := len(g.Dataset.Chars) + 3
	// The output vector representation is
	// 0 bit for the unknown character
	// [1,len(g.Dataset.Chars)] for characters
	// len(g.Dataset.Chars)+1 for the linefeed
	outputSize := len(g.Dataset.Chars) + 2

	input := make([][]float64, 0)
	for _, line := range poem {
		jndex := rand.Intn(len(line))
		for j, c := range line {
			cv := make([]float64, inputSize)
			if j == jndex {
				cv[c] = 1
			}
			input = append(input, cv)
		}

		linefeed := make([]float64, inputSize)
		linefeed[inputSize-2] = 1
		input = append(input, linefeed)
	}
	endOfPoem := make([]float64, inputSize)
	endOfPoem[inputSize-1] = 1
	input = append(input, endOfPoem)

	output := make([][]float64, 0)
	for range input {
		output = append(output, make([]float64, outputSize))
	}

	prevC := -1
	for _, line := range poem {
		for _, c := range line {
			cvi := make([]float64, inputSize)
			if prevC >= 0 {
				cvi[prevC] = 1
			}
			input = append(input, cvi)
			prevC = c
			cv := make([]float64, outputSize)
			cv[c] = 1
			output = append(output, cv)
		}

		cvi := make([]float64, inputSize)
		cvi[prevC] = 1
		input = append(input, cvi)
		prevC = inputSize - 2
		linefeed := make([]float64, outputSize)
		linefeed[outputSize-1] = 1
		output = append(output, linefeed)
	}

	return input, output
}

func (g *Generator) resample() {
	g.indices = rand.Perm(len(g.indices))
	g.offset = 0
}
