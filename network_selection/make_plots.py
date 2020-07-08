import plotly.graph_objects as go
import plotly.io as pio
from plotly.validators.scatter.marker import SymbolValidator
import plotly.express as px
#pio.templates.default = "plotly_dark"

# Add data



def make_plot(curves,name):
    #color_list = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2",  "#CC79A7"]
    budget = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    name_to_name={'sener_et_al':'Sener et al.',
                  'gradnorm': 'GradNorm',
                  'worst':'Worst Network Choice',
                  'all_in_one':'Single Traditional<br>Multi-task Network',
                  'random':'Random gropings',
                  'independent':'Five Independent Networks',
                  'esa':'ESA (ours) 5.3.1',
                  'hoa':'HOA (ours) 5.3.2',
                  'optimal':'Optimal Network<br>Choice (ours)' }

    name_to_color={'sener_et_al':7,
                   'gradnorm':8,
                   'worst':0,
                   'all_in_one':1,
                   'random':2,
                   'independent':3,
                   'esa':4,
                   'hoa':5,
                   'optimal':6}

    fig = go.Figure()

    symbols=['circle','square','diamond','star','hexagram','star-triangle-up','asterisk','y-up','cross']
    
    for i,(key,val) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=budget, y=val, name=name_to_name[key],connectgaps=True ,marker_symbol=name_to_color[key],marker_size=10,line=dict(color=px.colors.qualitative.G10[name_to_color[key]])))

    #line=dict(color=)
    
    # Create and style traces
    # if 'sener_et_al' in curves:
    #     fig.add_trace(go.Scatter(x=budget, y=curves['sener_et_al'], name='Sener et al.',connectgaps=True ,))
    # if 'gradnorm' in curves:
    #     fig.add_trace(go.Scatter(x=budget, y=curves['gradnorm'], name='GradNorm',connectgaps=True ))
    # fig.add_trace(go.Scatter(x=budget, y=curves['worst'], name='Worst Network<br>  Choice',connectgaps=True ))
    # fig.add_trace(go.Scatter(x=budget, y=curves['all_in_one'], name='Single Traditional<br>  Multi-task Network',connectgaps=True ))
    # fig.add_trace(go.Scatter(x=budget, y=curves['random'], name='Random Groupings',connectgaps=True ))
    # fig.add_trace(go.Scatter(x=budget, y=curves['independent'], name='Five Independent<br>  Networks',connectgaps=True ))
    # fig.add_trace(go.Scatter(x=budget, y=curves['esa'], name='ESA (ours) 3.3.1',connectgaps=True ))
    # fig.add_trace(go.Scatter(x=budget, y=curves['hoa'], name='HOA (ours) 3.3.2',connectgaps=True ))
    # fig.add_trace(go.Scatter(x=budget, y=curves['optimal'], name='Optimal Network<br>  Choice (ours)',connectgaps=True ))


    # Edit the layout
    fig.update_layout(title=dict(text='Performance vs Compute', font=dict(size=22,color='black')),
                    xaxis_title=dict(text='Inference Time Cost',font=dict(size=18,color='black')),
                    yaxis_title=dict(text='Total Loss (lower is better)',font=dict(size=18,color='black')),
                    legend=dict(font=dict(color='black',size=16)),
                    #colorway=px.colors.qualitative.G10,
                    xaxis=dict(
                        showline=True,
                        showgrid=False,
                        showticklabels=True,
                        linecolor='rgb(0, 0, 0)',
                        linewidth=1,
                        ticks='outside',
                        tickfont=dict(
                            family='Arial',
                            size=15,
                            color='rgb(0, 0, 0)',
                        ),
                    ),
                    yaxis=dict(
                        showgrid=True,
                        #zeroline=False,
                        ticks='outside',
                        showline=True,
                        showticklabels=True,
                        linecolor='rgb(0, 0, 0)',
                        linewidth=1,
                        tickfont=dict(
                            family='Arial',
                            size=15,
                            color='rgb(0, 0, 0)',
                        ),
                    ),
                    autosize=False,
                    margin=dict(
                        autoexpand=False,
                        l=58,
                        r=240,
                        t=32,
                        b=47
                    ),
                    width=600,
                    height=100+27*len(curves),
                    #showlegend=False,
                    plot_bgcolor='white'
                    )

    fig.write_image('plots/'+name+'.pdf')
    #fig.show()


curves_1=dict(
sener_et_al = [0.5621, None, 0.5556, None, None, None, 0.5471],
gradnorm = [0.5148, None, None, None, None, None, 0.5001],
worst = [0.50278, 0.50278, 0.50278, 0.50278, 0.50278, 0.50278, 0.50179, 0.50179, 0.49941],
all_in_one = [0.50273, None, 0.4916, 0.48873, None, None, 0.4883],
random = [0.50278, 0.485347, 0.473641, 0.469079, 0.465265, 0.46271, 0.460238, 0.458358, 0.456486],
independent = [0.51456, 0.50139, 0.47704, 0.46515, None, None, 0.45456, None, 0.44774],
esa = [0.50273, 0.48732, 0.46727, 0.46063, 0.45722, 0.45058, 0.45058, 0.44742, 0.44742],
hoa = [0.50278, 0.46132, 0.45474, 0.4505, 0.44875, 0.44489, 0.44112, 0.44552, 0.44196],
optimal = [0.50273, 0.46132, 0.45224, 0.44612, 0.44235, 0.43932, 0.43555, 0.43555, 0.43481],
)

curves_2=dict(
worst = [0.35989, 0.36554, 0.36926, 0.36936, 0.36956, 0.36956, 0.36956, 0.36956, 0.36956],
independent = [0.37276,0.35715,0.35926,0.36188,None,None,0.35384,None,0.35216],
all_in_one = [0.35989,None,0.35408,None,0.35431,None,0.35295,None,None],
random = [0.35989, 0.360109, 0.357285, 0.355924, 0.353176, 0.351664, 0.349508, 0.348102, 0.346303] ,
esa = [0.35989, 0.35989, 0.34696, 0.34696, 0.34483, 0.34483, 0.34483, 0.34483, 0.34483], 
hoa = [0.35989, 0.35758, 0.31733, 0.31562, 0.31177, 0.30525, 0.3019, 0.3019, 0.30187], 
optimal = [0.35989, 0.35478, 0.31733, 0.3145, 0.30606, 0.3049, 0.3019, 0.3019, 0.30167],
)

curves_3=dict(
worst = [0.42998, 0.47544, 0.47182, 0.47205, 0.4717, 0.47066, 0.46857, 0.46702, 0.46495] ,
all_in_one = [0.42998, None, None, None, None, None, 0.44391 ],
random = [0.42998, 0.439361, 0.439917, 0.435501, 0.431542, 0.427947, 0.424582, 0.421834, 0.419124],
independent = [0.41805, None,  None,  0.4262,  None,  None,  None, None, 0.40643],

esa = [0.42998, 0.44778, 0.43055, 0.39507, 0.40381, 0.39404, 0.40278, 0.39404, 0.40278] ,
hoa = [0.42998, 0.44778, 0.40887, 0.38776, 0.38352, 0.38682, 0.38574, 0.38574, 0.38471] ,
optimal = [0.42998, 0.42275, 0.40857, 0.38776, 0.38352, 0.38352, 0.38249, 0.38249, 0.38249],

)

curves_4=dict(
    worst = [0.684042, 0.689178, 0.696036, 0.698235, 0.700446, 0.701056, 0.701056, 0.701056, 0.701056],
    independent= [0.698867,None,None,0.692437,None,None,None,None,0.685578],
    random = [0.684042, 0.683817, 0.681984, 0.681949, 0.680581, 0.6801, 0.679037, 0.678471, 0.677633] ,
    
    
    esa = [0.684042, 0.684042, 0.677567, 0.680649, 0.677349, 0.676049, 0.676049, 0.675976, 0.675976] ,
    all_in_one = [0.684042,None,None,None,None,None,0.672991],
    
    hoa = [0.684042, 0.678697, 0.674597, 0.671067, 0.669696, 0.671867, 0.670496, 0.670496, 0.670496] ,
    optimal = [0.684042, 0.678697, 0.674597, 0.671067, 0.669696, 0.669696, 0.668986, 0.668986, 0.668986],
)


make_plot(curves_1,'setting_1')
make_plot(curves_2,'setting_2')
make_plot(curves_3,'setting_3')
make_plot(curves_4,'setting_4')