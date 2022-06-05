package main

import (
	"bufio"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
)

var (
	debugFlag   = flag.Bool("d", false, "print debug info")
	verboseFlag = flag.Bool("v", false, "print more info")
	equFileFlag = flag.String("file", "equ.txt", "the file containing the equations")
	minFlag     = flag.Bool("min", false, "determine whether min, if not min, then max is assumed")
)

type equElem struct {
	multiplier float32
	literal    string
}

type equation struct {
	vars []equElem
	RHS  float32
}

type Visitor struct {
	fset *token.FileSet
}

// func (v *Visitor) Visit(n ast.Node) ast.Visitor {
// 	if n == nil {
// 		return nil
// 	}

// 	switch x := n.(type) {
// 	case *ast.BinaryExpr:
// 		id := x.OpPos
// 		fmt.Println(id)
// 	case *ast.Ident:
// 		fmt.Println(x.Name)
// 	}
// 	return v
// }

// func parseEqu() error {
// 	fset := token.NewFileSet()
// 	file, err := parser.ParseFile(fset, *equFileFlag, nil, 0)
// 	if err != nil {
// 		fmt.Println(err)
// 		return err
// 	}
// 	fmt.Println("fuck")

// 	visitor := &Visitor{fset: fset}
// 	ast.Walk(visitor, file)

// 	return nil
// }

func walkExpr(expr *ast.BinaryExpr) *ast.BinaryExpr {
	if expr == nil {
		return nil
	}
	X := expr.X
	Y := expr.Y
	Op := expr.Op

	switch x := X.(type) {
	case *ast.Ident:
		if *debugFlag {
			fmt.Println("x.Name: ", x.Name)
		}
	case *ast.BasicLit:
		if *debugFlag {
			fmt.Println("x.Lit: ", x.Value)
		}
	case *ast.BinaryExpr:
		walkExpr(x)
	}

	switch y := Y.(type) {
	case *ast.Ident:
		if *debugFlag {
			fmt.Println("y.Name: ", y.Name)
		}
	case *ast.BasicLit:
		if *debugFlag {
			fmt.Println("y.Lit: ", y.Value)
		}
	case *ast.BinaryExpr:
		walkExpr(y)
	}

	if Op.IsOperator() {
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
		if ok {
			walkExpr(bExpr)
		}

		if *debugFlag {
			fmt.Println(fileScanner.Text())
			ast.Print(fs, exp)
		}
	}
	return nil
}

func main() {
	flag.Parse()
	err := parseInput()
	if err != nil {
		fmt.Println(err)
	}
}
