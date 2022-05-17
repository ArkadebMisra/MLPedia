from django import forms


scale_choices = (
    ("linear", "linear"),
    ("log", "log")
)

no_of_ele_choices = (
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5)
)

class LineGraphCreateForm(forms.Form):
    graph_title = forms.CharField(label='Graph Title', 
                                max_length=100,
                                required=False)

    x_axis_label = forms.CharField(label='X Axis Label',
                                max_length=100,
                                required=False)
    x_scale = forms.ChoiceField(label='X Axis Scale', choices = scale_choices)
    # x_lim_start = forms.IntegerField(label='X axis start')
    # x_lim_end = forms.IntegerField(label='X axis end')
    

    y_axis_label = forms.CharField(label='Y Axis Label',
                                max_length=100,
                                required=False)

    y_scale = forms.ChoiceField(label='Y Axis Scale', 
                                choices = scale_choices)
    # y_lim_start = forms.IntegerField(label='Y axis start')
    # y_lim_end = forms.IntegerField(label='Y axis end')

    number_of_lines = forms.ChoiceField(label='Number Of Lines', 
                    choices = no_of_ele_choices,)

    
    data1_label = forms.CharField(label='Data1 label',
                                required=False)
    x1 = forms.CharField(label = 'Line1 X values')
    y1 = forms.CharField(label = 'Line1 Y values')


    data2_label = forms.CharField(label='Data2 label',
                                required=False)
    x2 = forms.CharField(label = 'Line2 X values',
                        required=False)
    y2 = forms.CharField(label = 'Line2 Y values',
                        required=False)


    data3_label = forms.CharField(label='Data3 label',
                                required=False)
    x3 = forms.CharField(label = 'Line3 X values',
                        required=False)
    y3 = forms.CharField(label = 'Line3 Y values',
                        required=False)

    data4_label = forms.CharField(label='Data4 label',
                                required=False)
    x4 = forms.CharField(label = 'Line4 X values',
                        required=False)
    y4 = forms.CharField(label = 'Line4 Y values',
                        required=False)

    data5_label = forms.CharField(label='Data5 label',
                                required=False)
    x5 = forms.CharField(label = 'Line5 X values',
                        required=False)
    y5 = forms.CharField(label = 'Line5 Y values',
                        required=False)


bar_graph_oreantation_choices = (
    ("vertical", "vertical"),
    #("horizontal", "horizontal")
)

class BarGraphCreateForm(forms.Form):
    graph_title = forms.CharField(label='Graph Title', 
                                max_length=100,
                                required=False)

    # oreantation = forms.ChoiceField(label='Oreantation', 
    #                                 choices = bar_graph_oreantation_choices)
    x_axis_label = forms.CharField(label='X Axis Label',
                                max_length=100,
                                required=False)
    # x_lim_start = forms.IntegerField(label='X axis start')
    # x_lim_end = forms.IntegerField(label='X axis end')
    

    y_axis_label = forms.CharField(label='Y Axis Label',
                                max_length=100,
                                required=False)

    y_scale = forms.ChoiceField(label='Y Axis Scale', 
                                choices = scale_choices)
    # y_lim_start = forms.IntegerField(label='Y axis start')
    # y_lim_end = forms.IntegerField(label='Y axis end')

    number_of_bars = forms.ChoiceField(label='Number Of bars', 
                    choices = no_of_ele_choices,)

    
    data_labels = forms.CharField(label='Data labels',
                                required=True)
    bar1_label = forms.CharField(label = 'Bar 1 label')
    bar1_data_values = forms.CharField(label = 'Bar 1 Data Values')


    bar2_label = forms.CharField(label = 'Bar 2 label',
                                required=False)
    bar2_data_values = forms.CharField(label = 'Bar 2 Data Values',
                                required=False)

    bar3_label = forms.CharField(label = 'Bar 3 label',
                                required=False)
    bar3_data_values = forms.CharField(label = 'Bar 3 Data Values',
                                required=False)

    bar4_label = forms.CharField(label = 'Bar 4 label',
                                required=False)
    bar4_data_values = forms.CharField(label = 'Bar 4 Data Values',
                                required=False)

    bar5_label = forms.CharField(label = 'Bar 5 label',
                                required=False)
    bar5_data_values = forms.CharField(label = 'Bar 5 Data Values',
                                required=False)



class ScatterGraphCreateForm(forms.Form):
    graph_title = forms.CharField(label='Graph Title', 
                                max_length=100,
                                required=False)

    x_axis_label = forms.CharField(label='X Axis Label',
                                max_length=100,
                                required=False)
    x_scale = forms.ChoiceField(label='X Axis Scale', choices = scale_choices)
    # x_lim_start = forms.IntegerField(label='X axis start')
    # x_lim_end = forms.IntegerField(label='X axis end')
    

    y_axis_label = forms.CharField(label='Y Axis Label',
                                max_length=100,
                                required=False)

    y_scale = forms.ChoiceField(label='Y Axis Scale', 
                                choices = scale_choices)
    # y_lim_start = forms.IntegerField(label='Y axis start')
    # y_lim_end = forms.IntegerField(label='Y axis end')

    number_of_scatters = forms.ChoiceField(label='Number Of Scatters', 
                    choices = no_of_ele_choices,)

    
    data1_label = forms.CharField(label='Data1 label',
                                required=False)
    x1 = forms.CharField(label = 'Line1 X values')
    y1 = forms.CharField(label = 'Line1 Y values')


    data2_label = forms.CharField(label='Data2 label',
                                required=False)
    x2 = forms.CharField(label = 'Line2 X values',
                        required=False)
    y2 = forms.CharField(label = 'Line2 Y values',
                        required=False)


    data3_label = forms.CharField(label='Data3 label',
                                required=False)
    x3 = forms.CharField(label = 'Line3 X values',
                        required=False)
    y3 = forms.CharField(label = 'Line3 Y values',
                        required=False)

    data4_label = forms.CharField(label='Data4 label',
                                required=False)
    x4 = forms.CharField(label = 'Line4 X values',
                        required=False)
    y4 = forms.CharField(label = 'Line4 Y values',
                        required=False)

    data5_label = forms.CharField(label='Data5 label',
                                required=False)
    x5 = forms.CharField(label = 'Line5 X values',
                        required=False)
    y5 = forms.CharField(label = 'Line5 Y values',
                        required=False)


class PieGraphCreateForm(forms.Form):
    graph_title = forms.CharField(label='Graph Title', 
                                max_length=100,
                                required=False)

    data_labels = forms.CharField(label = 'data labels')
    data_values = forms.CharField(label = 'data values')