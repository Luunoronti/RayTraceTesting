using System;
using System.Diagnostics;
using System.Numerics;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;

namespace WinformsApp;

public partial class MainWindow : Window
{
    private readonly Stopwatch _fpsTimer = new Stopwatch();
    private readonly Stopwatch _frameTimer = new Stopwatch();
    private int _frameCount;
    private double _lastFrameTime;
    private Renderer? _renderer;
    private D3DHost? _host;

    private bool _forward;
    private bool _backward;
    private bool _left;
    private bool _right;
    private bool _up;
    private bool _down;
    private bool _fastMove;
    private bool _mouseLookActive;
    private Point _lastMousePos;
    private Vector2 _pendingLookDelta;

    public MainWindow()
    {
        InitializeComponent();

        Loaded += OnLoaded;
        Closing += OnClosing;
        CompositionTarget.Rendering += OnRendering;

        SppSlider.ValueChanged += (_, _) => ApplySettings();
        BouncesSlider.ValueChanged += (_, _) => ApplySettings();
        AmbientSlider.ValueChanged += (_, _) => ApplySettings();
        EmbossSlider.ValueChanged += (_, _) => ApplySettings();
        VsyncCheck.Checked += (_, _) => ApplySettings();
        VsyncCheck.Unchecked += (_, _) => ApplySettings();

        KeyDown += OnKeyDown;
        KeyUp += OnKeyUp;
    }

    private void OnLoaded(object? sender, RoutedEventArgs e)
    {
        _host = new D3DHost();
        RenderHost.Content = _host;
        RenderHost.Loaded += (_, _) => EnsureRenderer();
        RenderHost.SizeChanged += (_, _) => ResizeRenderer();
        InputOverlay.PreviewMouseRightButtonDown += OnMouseDown;
        InputOverlay.PreviewMouseRightButtonUp += OnMouseUp;
        InputOverlay.PreviewMouseMove += OnMouseMove;
        InputOverlay.Focusable = true;
        InputOverlay.Focus();

        _fpsTimer.Restart();
        _frameTimer.Restart();

        Dispatcher.BeginInvoke(DispatcherPriority.Loaded, new Action(EnsureRenderer));
    }

    private void OnClosing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        CompositionTarget.Rendering -= OnRendering;
        _renderer?.Dispose();
        _renderer = null;
    }

    private void OnRendering(object? sender, EventArgs e)
    {
        if (_renderer == null)
        {
            return;
        }

        double now = _frameTimer.Elapsed.TotalSeconds;
        float deltaTime = (float)(now - _lastFrameTime);
        _lastFrameTime = now;

        var moveInput = new Vector3(
            (_right ? 1.0f : 0.0f) - (_left ? 1.0f : 0.0f),
            (_up ? 1.0f : 0.0f) - (_down ? 1.0f : 0.0f),
            (_forward ? 1.0f : 0.0f) - (_backward ? 1.0f : 0.0f));

        if (_fastMove)
        {
            moveInput *= 3.0f;
        }

        var lookDelta = _pendingLookDelta;
        _pendingLookDelta = Vector2.Zero;

        _renderer.UpdateCamera(deltaTime, moveInput, lookDelta);
        _renderer.Render();
        TrackFps();
    }

    private void TrackFps()
    {
        _frameCount++;
        if (_fpsTimer.ElapsedMilliseconds >= 1000)
        {
            Title = $"3D Window (WPF) - FPS: {_frameCount}";
            _frameCount = 0;
            _fpsTimer.Restart();
        }
    }

    private void ApplySettings()
    {
        if (_renderer == null)
        {
            return;
        }

        _renderer.UpdateSettings(
            (int)SppSlider.Value,
            (int)BouncesSlider.Value,
            (float)AmbientSlider.Value,
            VsyncCheck.IsChecked == true,
            (float)EmbossSlider.Value);
    }

    private void ResizeRenderer()
    {
        if (_renderer == null)
        {
            EnsureRenderer();
            return;
        }

        int width = Math.Max(1, (int)RenderHost.ActualWidth);
        int height = Math.Max(1, (int)RenderHost.ActualHeight);
        _renderer.Resize(width, height);
    }

    private void EnsureRenderer()
    {
        if (_renderer != null || _host == null)
        {
            return;
        }

        if (_host.Handle == IntPtr.Zero)
        {
            return;
        }

        int width = Math.Max(1, (int)RenderHost.ActualWidth);
        int height = Math.Max(1, (int)RenderHost.ActualHeight);
        if (width == 0 || height == 0)
        {
            return;
        }

        _renderer = new Renderer(_host.Handle, width, height);
        ApplySettings();
    }

    private void OnKeyDown(object? sender, KeyEventArgs e)
    {
        switch (e.Key)
        {
            case Key.W:
                _forward = true;
                break;
            case Key.S:
                _backward = true;
                break;
            case Key.A:
                _left = true;
                break;
            case Key.D:
                _right = true;
                break;
            case Key.E:
                _up = true;
                break;
            case Key.Q:
                _down = true;
                break;
            case Key.LeftShift:
            case Key.RightShift:
                _fastMove = true;
                break;
        }
    }

    private void OnKeyUp(object? sender, KeyEventArgs e)
    {
        switch (e.Key)
        {
            case Key.W:
                _forward = false;
                break;
            case Key.S:
                _backward = false;
                break;
            case Key.A:
                _left = false;
                break;
            case Key.D:
                _right = false;
                break;
            case Key.E:
                _up = false;
                break;
            case Key.Q:
                _down = false;
                break;
            case Key.LeftShift:
            case Key.RightShift:
                _fastMove = false;
                break;
        }
    }

    private void OnMouseDown(object? sender, MouseButtonEventArgs e)
    {
        if (e.ChangedButton == MouseButton.Right)
        {
            _mouseLookActive = true;
            _lastMousePos = e.GetPosition(InputOverlay);
            InputOverlay.CaptureMouse();
        }
    }

    private void OnMouseUp(object? sender, MouseButtonEventArgs e)
    {
        if (e.ChangedButton == MouseButton.Right)
        {
            _mouseLookActive = false;
            InputOverlay.ReleaseMouseCapture();
        }
    }

    private void OnMouseMove(object? sender, MouseEventArgs e)
    {
        if (!_mouseLookActive)
        {
            return;
        }

        Point pos = e.GetPosition(InputOverlay);
        double dx = pos.X - _lastMousePos.X;
        double dy = pos.Y - _lastMousePos.Y;
        _pendingLookDelta += new Vector2((float)dx, (float)dy);
        _lastMousePos = pos;
    }
}
