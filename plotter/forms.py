from django import forms


scale_choices = (
    ("linear", "linear"),
    ("log", "log")
)

no_of_line_choices = (
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5)
)

class LineGraphCreateForm(forms.Form):
    graph_title = forms.CharField(label='Graph Title', max_length=100)

    x_axis_label = forms.CharField(label='X Axis Label', max_length=100)
    x_scale = forms.ChoiceField(label='X Axis Scale', choices = scale_choices)
    # x_lim_start = forms.IntegerField(label='X axis start')
    # x_lim_end = forms.IntegerField(label='X axis end')
    

    y_axis_label = forms.CharField(label='Y Axis Label', max_length=100)
    y_scale = forms.ChoiceField(label='Y Axis Scale', choices = scale_choices)
    # y_lim_start = forms.IntegerField(label='Y axis start')
    # y_lim_end = forms.IntegerField(label='Y axis end')

    number_of_lines = forms.ChoiceField(label='Number Of Lines', 
                    choices = no_of_line_choices,
                    initial=None)

    
    x1 = forms.CharField(label = 'Line1 X values')
    y1 = forms.CharField(label = 'Line1 Y values')

    x2 = forms.CharField(label = 'Line2 X values')
    y2 = forms.CharField(label = 'Line2 Y values')
