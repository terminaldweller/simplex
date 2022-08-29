#!/usr/bin/env python3
"""dsimplex gui"""

import tkinter as tk
import tkinter.filedialog
import os
from .args import Argparser
from .simplex import dsimplex_gui


class DsimplexGui:
    """The GUI class."""

    def __init__(self) -> None:
        self.is_minimization: bool = True
        self.csv_file_path: str = ""
        self.argparse: Argparser = Argparser()
        self.mock_cli: str = ""
        self.html_report_dir: str = ""

        self.window = tk.Tk()
        self.window.title("dsimplex")

        # self.aux_var_name: tk.StringVar = tk.StringVar(self.window, value="xa")
        # self.slack_var_name: tk.StringVar = tk.StringVar(
        #     self.window, value="s"
        # )
        # self.csv_delim: tk.StringVar = tk.StringVar(self.window, value=",")

        self.window.rowconfigure(0, minsize=800, weight=1)
        self.window.columnconfigure(0, minsize=800, weight=1)

        self.label = tk.Label(text="dsimplex", foreground="black")
        self.label.pack()

        self.frame_left = tk.Frame(relief=tk.GROOVE, borderwidth=3)
        self.frame_right = tk.Frame(relief=tk.GROOVE, borderwidth=3)

        self.checkbutton_min = tk.Checkbutton(
            self.window,
            text="is minimization:",
            variable=self.is_minimization,
            onvalue=True,
        )
        self.checkbutton_min.pack()

        # self.label_aux_var_name = tk.Label(
        #     text="auxillary var name:", textvariable=self.aux_var_name
        # )
        self.label_aux_var_name = tk.Label(text="auxillary var name:")
        self.label_aux_var_name.pack()
        self.entry_aux_var_name = tk.Entry()
        self.entry_aux_var_name.insert(tk.END, "xa")
        self.entry_aux_var_name.pack()

        # self.label_slack_var_name = tk.Label(
        #     text="slack var name:", textvariable=self.slack_var_name
        # )
        self.label_slack_var_name = tk.Label(text="slack var name:")
        self.label_slack_var_name.pack()
        self.entry_slack_var_name = tk.Entry()
        self.entry_slack_var_name.insert(tk.END, "s")
        self.entry_slack_var_name.pack()

        # self.label_csv_delim_ = tk.Label(
        #     text="CSV delimiter:", textvariable=self.csv_delim
        # )
        self.label_csv_delim_ = tk.Label(text="CSV delimiter:")
        self.label_csv_delim_.pack()
        self.entry_csv_delim = tk.Entry()
        self.entry_csv_delim.insert(tk.END, ",")
        self.entry_csv_delim.pack()

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

        self.button_browse.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.button_run.grid(row=0, column=4, sticky="ew", padx=5)
        self.button_html_report_dir.grid(
            row=0, column=1, sticky="ew", padx=5, pady=5
        )

        self.text = tk.Text()
        self.text.pack()
        self.text.configure(state="normal")

        self.frame_right.pack()
        self.frame_left.pack()

    def open_file_browser_cb(self) -> None:
        """Callback function for the file browser button."""
        csv_file = tk.filedialog.askopenfile(mode="r")
        if csv_file:
            self.csv_file_path = os.path.abspath(csv_file.name)
            # content = csv_file.read()
            # csv_file.close()
            # self.text.insert(tk.END, content)

    def open_output_browser_cb(self) -> None:
        """Callabck for the button to select the html report dir."""
        html_dir = tk.filedialog.askdirectory()
        if html_dir:
            self.html_report_dir = os.path.abspath(html_dir)

    def run_button_cb(self) -> None:
        """Callback for the run button."""
        if self.csv_file_path == "":
            return
        self.mock_cli += " --delim " + self.entry_csv_delim.get() + " "
        self.mock_cli += " --csv " + self.csv_file_path + " "
        if self.is_minimization:
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
        result = dsimplex_gui(self.argparse)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, repr(result))

    def main_loop(self) -> None:
        """Runs the main loop."""
        self.window.mainloop()

    def mock_argparser(self):
        """Generates a dummy argparser holding the args."""
