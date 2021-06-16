# -*- coding: utf-8 -*-

def print_free_style(message, print_fun=print):
    print_fun("▓  {}".format(message))
    print_fun("")

def print_time_style(message, print_fun=print):
    print_fun("")
    print_fun("⏰  {}".format(message))
    print_fun("")
    
def print_warning_style(message, print_fun=print):
    print_fun("")
    print_fun("⛔️  {}".format(message))
    print_fun("")
    
def print_notice_style(message, print_fun=print):
    print_fun("")
    print_fun("📌  {}".format(message))
    print_fun("")

def print_line(text, print_fun=print):
    print_fun("")
    print_fun("➖➖➖ {} ➖➖➖".format(text.upper()))
    print_fun("")
