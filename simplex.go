package main

import (
	"bufio"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

var (
	debugFlag   = flag.Bool("d", false, "print debug info")
	verboseFlag = flag.Bool("v", false, "print more info")
	equFileFlag = flag.String("file", "equ.txt", "the file containing the equations")
	minFlag     = flag.Bool("min", false, "determine whether min, if not min, then max is assumed")
)

type equElem struct {
	multiplier float64
	literal    string
}

type equation struct {
	vars []equElem
	Op   token.Token
	RHS  float64
}

func walkExpr(expr *ast.BinaryExpr, equ *equation) *ast.BinaryExpr {
	if expr == nil {
		return nil
	}
	var err error
	X := expr.X
	Y := expr.Y
	Op := expr.Op

	var tmp_equ_x equElem
	equ.vars = append(equ.vars, tmp_equ_x)
	switch x := X.(type) {
	case *ast.Ident:
		equ.vars[0].literal = x.Name
		if *debugFlag {
			fmt.Println("x.Name: ", x.Name)
		}
	case *ast.BasicLit:
		equ.vars[0].multiplier, err = strconv.ParseFloat(x.Value, 64)
		if err != nil {
			log.Fatal(err)
		}
		if *debugFlag {
			fmt.Println("x.Lit: ", x.Value)
		}
	case *ast.BinaryExpr:
		walkExpr(x, equ)
	}

	var tmp_equ_y equElem
	equ.vars = append(equ.vars, tmp_equ_y)
	switch y := Y.(type) {
	case *ast.Ident:
		equ.vars[1].literal = y.Name
		if *debugFlag {
			fmt.Println("y.Name: ", y.Name)
		}
	case *ast.BasicLit:
		equ.vars[1].multiplier, err = strconv.ParseFloat(y.Value, 64)
		if err != nil {
			log.Fatal(err)
		}
		if *debugFlag {
			fmt.Println("y.Lit: ", y.Value)
		}
	case *ast.BinaryExpr:
		walkExpr(y, equ)
	}

	if Op.IsOperator() {
		equ.Op = Op
		if *debugFlag {
			fmt.Println("Op: ", Op.String())
		}
	}
	if *debugFlag {
		fmt.Println("\n")
	}

	return nil
}

func parseInput() error {
	var equations []equation
	readFile, err := os.Open(*equFileFlag)
	if err != nil {
		return nil
	}
	defer readFile.Close()

	fs := token.NewFileSet()
	fileScanner := bufio.NewScanner(readFile)
	fileScanner.Split(bufio.ScanLines)
	for fileScanner.Scan() {
		exp, err := parser.ParseExpr(fileScanner.Text())
		if err != nil {
			return err
		}

		bExpr, ok := exp.(*ast.BinaryExpr)
		var equ equation
		if ok {
			walkExpr(bExpr, &equ)
		}
		equations = append(equations, equ)

		if *debugFlag {
			fmt.Println(fileScanner.Text())
			ast.Print(fs, exp)
		}
	}
	return nil
}

func addSlackVars(A *mat.Dense) error {

	return nil
}

func getBasis(A, b *mat.Dense) *mat.Dense {
	allPositive := true
	brow, _ := b.Dims()
	B := mat.NewDense(brow, brow, nil)
	for i := 0; i < brow; i++ {
		if b.At(i, 0) < 0 {
			allPositive = false
		}
	}

	if allPositive {
		for i := 0; i < brow; i++ {
			B.Set(i, i, 1)
		}
		return B
	}

	return B
}

func doOneRound(A, B, b, Z *mat.Dense) error {
	var BInv *mat.Dense
	err := BInv.Inverse(B)
	if err != nil {
		return err
	}

	var bbar *mat.Dense
	bbar.Mul(BInv, b)

	return nil
}

func main() {
	flag.Parse()
	err := parseInput()
	if err != nil {
		fmt.Println(err)
	}
}
