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
	verboseFlag = flag.Bool("v", false, "print more info")
	equFileFlag = flag.String("file", "equ.txt", "the file containing the equations")
	minFlag     = flag.Bool("min", false, "determine whether min, if not min, then max is assumed")
)

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
		tr, err := parser.ParseExpr(fileScanner.Text())
		if err != nil {
			return err
		}
		if *verboseFlag {
			fmt.Println(fileScanner.Text())
			ast.Print(fs, tr)
		}
	}
	return nil
}

func main() {
	flag.Parse()
	parseInput()
}
