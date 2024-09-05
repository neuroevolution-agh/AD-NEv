import xlrd
from xlrd.timemachine import xrange


def read_data_xlsx(filenames, start_index = 1):
    dict_list = []
    for f in filenames:
        data = xlrd.open_workbook(f)
        sheet = data.sheet_by_index(0)
        keys = [sheet.cell(0, col_index).value for col_index in xrange(sheet.ncols)]
        keys = [x.strip(' ') for x in keys]
        for row_index in xrange(start_index, sheet.nrows):
            d = {keys[col_index]: sheet.cell(row_index, col_index).value for col_index in xrange(sheet.ncols)}
            dict_list.append(d)
    return dict_list
