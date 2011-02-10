import wx
import string

class Validator(wx.PyValidator):
    def __init__(self, flag):
        wx.PyValidator.__init__(self)
        self.flag = flag
        self.Bind(wx.EVT_CHAR, self.OnChar)
    def Clone(self):
        return Validator(self.flag)
    def Validate(self, win):
        return True
    def TransferToWindow(self):
        return True
    def TransferFromWindow(self):
        return True
    def OnChar(self, evt):
        key = chr(evt.GetKeyCode())
        if self.flag == "no-alpha" and key in string.letters:
            return
        if self.flag == "no-digit" and key in string.digits:
            return
        evt.Skip()

