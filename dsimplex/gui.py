#!/usr/bin/env python3
"""dsimplex gui"""

import os
import tkinter as tk
import tkinter.filedialog

import markdown
import tk_html_widgets

from .args import Argparser
from .simplex import dsimplex_gui, parse_equ_csv_loop


class DsimplexGui:
    """The GUI class."""

    def __init__(self) -> None:
        self.csv_file_path: str = ""
        self.argparse: Argparser = Argparser()
        self.mock_cli: str = ""
        self.html_report_dir: str = ""

        self.window = tk.Tk()
        self.window.title("dsimplex")
        self.is_minimization: tk.BooleanVar = tk.BooleanVar()

        self.window.rowconfigure(0, minsize=800, weight=1)
        self.window.columnconfigure(0, minsize=800, weight=1)

        self.label = tk.Label(text="dsimplex", foreground="black")
        self.label.pack()

        self.frame_left = tk.Frame(relief=tk.GROOVE, borderwidth=3)
        self.frame_right = tk.Frame(relief=tk.GROOVE, borderwidth=3)

        self.checkbutton_min = tk.Checkbutton(
            self.window, text="is minimization:", variable=self.is_minimization
        )
        self.checkbutton_min.pack()

        self.label_aux_var_name = tk.Label(text="auxillary var name:")
        self.label_aux_var_name.pack()
        self.entry_aux_var_name = tk.Entry()
        self.entry_aux_var_name.insert(tk.END, "xa")
        self.entry_aux_var_name.pack()

        self.label_slack_var_name = tk.Label(text="slack var name:")
        self.label_slack_var_name.pack()
        self.entry_slack_var_name = tk.Entry()
        self.entry_slack_var_name.insert(tk.END, "s")
        self.entry_slack_var_name.pack()

        self.label_csv_delim_ = tk.Label(text="CSV delimiter:")
        self.label_csv_delim_.pack()
        self.entry_csv_delim = tk.Entry()
        self.entry_csv_delim.insert(tk.END, ",")
        self.entry_csv_delim.pack()

        self.label_max_iter = tk.Label(text="Maximum iterations:")
        self.label_max_iter.pack()
        self.entry_max_iter = tk.Entry()
        self.entry_max_iter.insert(tk.END, "50")
        self.entry_max_iter.pack()

        self.button_run = tk.Button(
            text="run",
            width=10,
            height=2,
            master=self.frame_left,
            command=self.run_button_cb,
        )

        self.button_browse = tk.Button(
            text="select CSV file",
            width=10,
            height=2,
            command=self.open_file_browser_cb,
            master=self.frame_left,
        )

        self.button_html_report_dir = tk.Button(
            text="HTML report path",
            width=10,
            height=2,
            master=self.frame_left,
            command=self.open_output_browser_cb,
        )

        self.button_help = tk.Button(
            text="Help",
            width=10,
            height=2,
            master=self.frame_left,
            command=self.open_help_window,
        )

        self.button_show_report = tk.Button(
            text="Show Report",
            width=10,
            height=2,
            master=self.frame_left,
            command=self.show_report_cb,
        )

        self.button_browse.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.button_run.grid(row=0, column=4, sticky="ew", padx=5)
        self.button_html_report_dir.grid(
            row=0, column=1, sticky="ew", padx=5, pady=5
        )
        self.button_help.grid(row=0, column=5, sticky="ew", padx=5, pady=5)
        self.button_show_report.grid(
            row=0, column=6, sticky="ew", padx=5, pady=6
        )

        self.text = tk.Text()
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="normal")

        self.frame_right.pack()
        self.frame_left.pack()

    def show_report_cb(self) -> None:
        """Callback for the show report button. displays the report."""
        report_window = tk.Toplevel()
        report_window.wm_title("dsimplex report")

        report_path: str = ""
        if self.html_report_dir != "":
            report_path = self.html_report_dir + "/dsimplex_report.html"
        else:
            report_path = os.getcwd() + "/dsimplex_report.html"

        with open(report_path, encoding="utf-8") as report_file:
            html_help_content = report_file.read()

            help_label = tk_html_widgets.HTMLScrolledText(report_window)
            help_label.set_html(html_help_content)

            help_label.pack(fil="both", expand=True)

    def open_file_browser_cb(self) -> None:
        """Callback function for the file browser button."""
        csv_file = tk.filedialog.askopenfile(mode="r")
        if csv_file:
            self.csv_file_path = os.path.abspath(csv_file.name)

    def open_help_window(self):
        """Callback for the help button."""
        help_window = tk.Toplevel()
        help_window.wm_title("dsimplex help")

        with open("README.md", encoding="utf-8") as help_file:
            md_help_content = help_file.read()
            html_help_content = markdown.markdown(md_help_content)

            help_label = tk_html_widgets.HTMLScrolledText(help_window)
            help_label.set_html(html_help_content)

            help_label.pack(fil="both", expand=True)

    def open_output_browser_cb(self) -> None:
        """Callabck for the button to select the html report dir."""
        html_dir = tk.filedialog.askdirectory()
        if html_dir:
            self.html_report_dir = os.path.abspath(html_dir)

    def run_button_cb(self) -> None:
        """Callback for the run button."""
        try:
            if self.csv_file_path == "":
                return
            self.mock_cli += " --delim " + self.entry_csv_delim.get() + " "
            self.mock_cli += " --csv " + self.csv_file_path + " "
            self.mock_cli += " --iter " + self.entry_max_iter.get() + " "
            if self.is_minimization.get():
                self.mock_cli += " -m "

            if self.html_report_dir != "":
                self.mock_cli += (
                    " --out "
                    + self.html_report_dir
                    + "/dsimplex_report.html"
                    + " "
                )
            else:
                self.mock_cli += (
                    " --out " + os.getcwd() + "/dsimplex_report.html" + " "
                )

            print(self.mock_cli)
            self.argparse.parse(self.mock_cli.split())
            result = parse_equ_csv_loop(self.argparse)
            print(result)
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, repr(result))
        except Exception as e:
            # we really don't care what the problem is. we just don't
            # want to exit the gui
            print(e)

    def main_loop(self) -> None:
        """Runs the main loop."""
        self.window.mainloop()

    def mock_argparser(self):
        """Generates a dummy argparser holding the args."""
