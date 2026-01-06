using System;
using System.Runtime.InteropServices;
using System.Windows.Interop;

namespace WinformsApp;

internal sealed class D3DHost : HwndHost
{
    private IntPtr _hwnd;

    public new IntPtr Handle => _hwnd;

    protected override HandleRef BuildWindowCore(HandleRef hwndParent)
    {
        const int wsChild = 0x40000000;
        const int wsVisible = 0x10000000;

        _hwnd = CreateWindowEx(
            0,
            "STATIC",
            string.Empty,
            wsChild | wsVisible,
            0,
            0,
            1,
            1,
            hwndParent.Handle,
            IntPtr.Zero,
            IntPtr.Zero,
            IntPtr.Zero);

        return new HandleRef(this, _hwnd);
    }

    protected override void DestroyWindowCore(HandleRef hwnd)
    {
        if (_hwnd != IntPtr.Zero)
        {
            DestroyWindow(_hwnd);
            _hwnd = IntPtr.Zero;
        }
    }

    [DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern IntPtr CreateWindowEx(
        int exStyle,
        string className,
        string windowName,
        int style,
        int x,
        int y,
        int width,
        int height,
        IntPtr parent,
        IntPtr menu,
        IntPtr instance,
        IntPtr param);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool DestroyWindow(IntPtr hwnd);
}
