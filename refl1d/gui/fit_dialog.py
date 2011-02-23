#!/usr/bin/python

# Find/Replace Dialog

import wx

class FindReplace(wx.Dialog):
    def __init__(self, parent, id, title):
        wx.Dialog.__init__(self, parent, id, title, size=(450, 445))

        vbox_top = wx.BoxSizer(wx.VERTICAL)
        panel = wx.Panel(self, -1)

        vbox = wx.BoxSizer(wx.VERTICAL)
        
        vbox.Add((-1, 15))

        # panel1

        panel1 = wx.Panel(panel, -1)
        
        vbox1 = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        
        save_label = wx.StaticText(panel1,  -1, 'Save location: ')
        self.save = wx.TextCtrl(panel1, -1, '',  size=(200, -1), style=wx.TE_RIGHT)  
        self.overwrite = wx.CheckBox(panel1, wx.ID_ANY, 'overwrite')
        hbox1.Add(save_label, 0,  wx.ALIGN_CENTER_VERTICAL)
        hbox1.Add(self.save)
        vbox1.Add(hbox1)
        vbox1.Add(self.overwrite)

        panel1.SetSizer(vbox1)
        vbox.Add(panel1, 0, wx.BOTTOM | wx.TOP, 9)

        vbox.Add((-1, 15))

        # panel2

        panel2 = wx.Panel(panel, -1)
        hbox2 = wx.BoxSizer(wx.VERTICAL)
        
        static_box1 = wx.StaticBox(panel2, -1, 'Fit Algorithms')

        ameoba_radio = wx.RadioButton(panel2, -1, 'Ameoba')
        de_radio = wx.RadioButton(panel2, -1, 'DE')
        dream_radio = wx.RadioButton(panel2, -1, 'Dream')
        pt_radio = wx.RadioButton(panel2, -1, 'Parallel Tempering')
        rl_radio = wx.RadioButton(panel2, -1, 'Random Lines')
        
        fit_hsizer = wx.StaticBoxSizer(static_box1, orient=wx.HORIZONTAL)
        fit_hsizer.Add(ameoba_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(de_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(dream_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(pt_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(rl_radio, 0, wx.ALL, 5)   
        
        hbox2.Add(fit_hsizer, 1, wx.RIGHT, 5)
        panel2.SetSizer(hbox2)
        vbox.Add(panel2, 0)
                
        vbox.Add((-1, 15)) 
        
        # panel3

        panel3 = wx.Panel(panel, -1)
                
        static_box2 = wx.StaticBox(panel3, -1, 'Plot Options')
        fresnel_radio = wx.RadioButton(panel3, -1, 'Fresnel')
        linear_radio = wx.RadioButton(panel3, -1, 'Linear')
        log_radio = wx.RadioButton(panel3, -1, 'Log')
        q4_radio = wx.RadioButton(panel3, -1, 'Q4')
        
        plot_hsizer = wx.StaticBoxSizer(static_box2, orient=wx.HORIZONTAL)
        plot_hsizer.Add(fresnel_radio, 0, wx.ALL, 5)
        plot_hsizer.Add(linear_radio, 0, wx.ALL, 5)        
        plot_hsizer.Add(log_radio, 0, wx.ALL, 5)
        plot_hsizer.Add(q4_radio, 0, wx.ALL, 5) 

        panel3.SetSizer(plot_hsizer)
        vbox.Add(panel3, 0, wx.BOTTOM, 0)

        vbox.Add((-1, 15))
        # panel4

        panel4 = wx.Panel(panel, -1)
        static_box3 = wx.StaticBox(panel4, -1, 'Other Options')
        other_hsizer = wx.StaticBoxSizer(static_box3, orient=wx.HORIZONTAL)
        vbox4 = wx.BoxSizer(wx.VERTICAL)
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer4 = wx.BoxSizer(wx.HORIZONTAL)
        
        noise_label = wx.StaticText(panel4,  -1, 'Noise: ')
        self.noise = wx.TextCtrl(panel4, -1, '', size=(150, -1), style=wx.TE_RIGHT)
        stepsize_label = wx.StaticText(panel4,  -1, 'Step size: ')
        self.stepsize = wx.TextCtrl(panel4, -1, '', size=(150, -1), style=wx.TE_RIGHT)
        burn_label = wx.StaticText(panel4,  -1, 'Burn: ')
        self.burn = wx.TextCtrl(panel4, -1, '', size=(150, -1), style=wx.TE_RIGHT)
        tmin_label = wx.StaticText(panel4,  -1, 'T min: ')
        self.tmin = wx.TextCtrl(panel4, -1, '',  size=(150, -1),style=wx.TE_RIGHT)
        tmax_label = wx.StaticText(panel4,  -1, 'T max: ')
        self.tmax = wx.TextCtrl(panel4, -1, '',  size=(150, -1),style=wx.TE_RIGHT)
        crossover_label = wx.StaticText(panel4,  -1, 'Crossover Ratio: ')
        self.crossover = wx.TextCtrl(panel4, -1, '', size=(150, -1), style=wx.TE_RIGHT)
        pop_label = wx.StaticText(panel4,  -1, 'Population: ')
        self.pop = wx.TextCtrl(panel4, -1, '', size=(150, -1), style=wx.TE_RIGHT)
        
        sizer1.Add(noise_label,0, wx.ALL, 5)
        sizer1.Add(self.noise,0, wx.ALL, 5)        
        sizer1.Add(stepsize_label,0, wx.ALL, 5)
        sizer1.Add(self.stepsize,0, wx.ALL, 5)        
        
        sizer2.Add(burn_label,0, wx.ALL, 5)
        sizer2.Add(self.burn,0, wx.ALL, 5)        
        sizer2.Add(pop_label,0, wx.ALL, 5)
        sizer2.Add(self.pop,0, wx.ALL, 5)        

        sizer3.Add(tmin_label,0, wx.ALL, 5)
        sizer3.Add(self.tmin,0, wx.ALL, 5)        
        sizer3.Add(tmax_label,0, wx.ALL, 5)
        sizer3.Add(self.tmax,0, wx.ALL, 5)        
        
        sizer4.Add(crossover_label,0, wx.ALL, 5)
        sizer4.Add(self.crossover,0, wx.ALL, 5)        
        
        vbox4.Add(sizer1)
        vbox4.Add(sizer2)
        vbox4.Add(sizer3)
        vbox4.Add(sizer4)                                
                
        other_hsizer.Add(vbox4, 0, wx.TOP, 4)

        panel4.SetSizer(other_hsizer)
        vbox.Add(panel4, 0, wx.BOTTOM, 15)
              
        vbox.Add((-1, 15))
        # panel5

        panel5 = wx.Panel(panel, -1)
        sizer5 = wx.BoxSizer(wx.HORIZONTAL)
        sizer5.Add((135, -1), 1, wx.EXPAND | wx.ALIGN_RIGHT)
        ok_btn = wx.Button(panel5, -1, 'Ok', size=(50, -1))
        cancel_btn = wx.Button(panel5, -1, 'Cancel', size=(50, -1))
        
        sizer5.Add(ok_btn)
        sizer5.Add(cancel_btn)

        panel5.SetSizer(sizer5)
        vbox.Add(panel5, 1, wx.BOTTOM, 9)

        vbox_top.Add(vbox, 1, wx.LEFT, 5)
        panel.SetSizer(vbox_top)

        self.Centre()
        self.ShowModal()
        self.Destroy()


app = wx.App()
FindReplace(None, -1, 'Find/Replace')
app.MainLoop()

