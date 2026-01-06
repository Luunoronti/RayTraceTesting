using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Threading;
using SharpGen.Runtime;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.Dxc;
using Vortice.DXGI;
using static Vortice.Direct3D12.D3D12;
using static Vortice.DXGI.DXGI;

namespace WinformsApp;

internal sealed class Renderer : IDisposable
{
    private const int FrameCount = 2;
    private const int DescriptorCount = 24;
    private const int CbvIndex = 0;
    private const int SrvBaseIndex = 1;
    private const int SrvBufferCount = 5;
    private const int SrvTextureCount = 7;
    private const int SrvGridCount = 3;
    private const int UavBaseIndex = SrvBaseIndex + SrvBufferCount + SrvTextureCount + SrvGridCount;

    private readonly IntPtr _hwnd;
    private readonly ID3D12Device _device;
    private readonly ID3D12CommandQueue _commandQueue;
    private readonly ID3D12CommandAllocator _commandAllocator;
    private readonly ID3D12GraphicsCommandList _commandList;
    private readonly IDXGISwapChain3 _swapChain;
    private readonly ID3D12Fence _fence;
    private readonly AutoResetEvent _fenceEvent;

    private readonly ID3D12Resource[] _backBuffers = new ID3D12Resource[FrameCount];
    private readonly ulong[] _frameFenceValues = new ulong[FrameCount];
    private int _frameIndex;
    private ulong _fenceValue;
    private int _width;
    private int _height;

    private ID3D12RootSignature _computeRootSignature = null!;
    private ID3D12PipelineState _rayPipelineState = null!;
    private ID3D12PipelineState _temporalPipelineState = null!;
    private ID3D12PipelineState _atrousPipelineState = null!;
    private ID3D12DescriptorHeap _uavHeap = null!;
    private int _cbvUavDescriptorSize;
    private ID3D12Resource _computeTexture = null!;
    private ID3D12Resource _constantBuffer = null!;
    private int _constantBufferSize;
    private ID3D12Resource _sphereBuffer = null!;
    private ID3D12Resource _planeBuffer = null!;
    private ID3D12Resource _boxBuffer = null!;
    private ID3D12Resource _materialBuffer = null!;
    private ID3D12Resource _lightBuffer = null!;
    private ID3D12Resource _gridCellStartBuffer = null!;
    private ID3D12Resource _gridCellCountBuffer = null!;
    private ID3D12Resource _gridIndexBuffer = null!;
    private int _sphereCount;
    private int _planeCount;
    private int _boxCount;
    private int _lightCount;
    private int _gridDimX;
    private int _gridDimY;
    private int _gridDimZ;
    private Vector3 _gridMin;
    private Vector3 _gridMax;
    private int _gridIndexCount;
    private Vector3 _cameraPosition = new Vector3(0, 0, -5.0f);
    private float _cameraYaw;
    private float _cameraPitch;
    private float _cameraFov = 60.0f;
    private Vector3 _cameraForward = Vector3.UnitZ;
    private Vector3 _cameraRight = Vector3.UnitX;
    private Vector3 _cameraUp = Vector3.UnitY;
    private ID3D12Resource _accumTexture = null!;
    private ID3D12Resource _normalTexture = null!;
    private ID3D12Resource _depthTexture = null!;
    private ID3D12Resource _historyTextureA = null!;
    private ID3D12Resource _historyTextureB = null!;
    private ID3D12Resource _momentTextureA = null!;
    private ID3D12Resource _momentTextureB = null!;
    private ResourceStates _computeState = ResourceStates.UnorderedAccess;
    private ResourceStates _accumState = ResourceStates.UnorderedAccess;
    private ResourceStates _normalState = ResourceStates.UnorderedAccess;
    private ResourceStates _depthState = ResourceStates.UnorderedAccess;
    private ResourceStates _historyAState = ResourceStates.UnorderedAccess;
    private ResourceStates _historyBState = ResourceStates.UnorderedAccess;
    private ResourceStates _momentAState = ResourceStates.UnorderedAccess;
    private ResourceStates _momentBState = ResourceStates.UnorderedAccess;
    private int _accumulationFrame;
    private int _samplesPerPixel = 2;
    private int _maxBounces = 3;
    private float _ambientStrength = 0.03f;
    private bool _vSync = true;
    private readonly Stopwatch _clock = Stopwatch.StartNew();

    public Renderer(IntPtr hwnd, int width, int height)
    {
        _hwnd = hwnd;
        _width = Math.Max(width, 1);
        _height = Math.Max(height, 1);

        using var factory = CreateDXGIFactory2<IDXGIFactory4>(false);
        using var adapter = PickAdapter(factory);

        _device = D3D12CreateDevice<ID3D12Device>(adapter, FeatureLevel.Level_12_0);
        _commandQueue = _device.CreateCommandQueue(new CommandQueueDescription(CommandListType.Direct));

        var swapDesc = new SwapChainDescription1
        {
            Width = _width,
            Height = _height,
            Format = Format.R8G8B8A8_UNorm,
            BufferCount = FrameCount,
            BufferUsage = Usage.RenderTargetOutput,
            SwapEffect = SwapEffect.FlipDiscard,
            SampleDescription = new SampleDescription(1, 0)
        };

        using var swapChain1 = factory.CreateSwapChainForHwnd(_commandQueue, _hwnd, swapDesc);
        _swapChain = swapChain1.QueryInterface<IDXGISwapChain3>();
        _frameIndex = _swapChain.CurrentBackBufferIndex;

        _commandAllocator = _device.CreateCommandAllocator<ID3D12CommandAllocator>(CommandListType.Direct);
        _commandList = _device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, _commandAllocator, null);
        _commandList.Close();

        _fence = _device.CreateFence(0);
        _fenceEvent = new AutoResetEvent(false);

        CreateBackBuffers();
        CreateComputeResources();
    }

    public void Render()
    {
        _commandAllocator.Reset();
        _commandList.Reset(_commandAllocator, null);

        var backBuffer = _backBuffers[_frameIndex];
        _commandList.ResourceBarrier(ResourceBarrier.BarrierTransition(
            backBuffer,
            ResourceStates.Present,
            ResourceStates.CopyDestination));

        UpdateComputeConstants();
        _commandList.SetDescriptorHeaps(1, new[] { _uavHeap });
        _commandList.SetComputeRootSignature(_computeRootSignature);
        _commandList.SetComputeRootDescriptorTable(0, _uavHeap.GetGPUDescriptorHandleForHeapStart());

        bool writeToA = (_accumulationFrame & 1) == 0;
        ID3D12Resource historyPrev = writeToA ? _historyTextureB : _historyTextureA;
        ID3D12Resource historyOut = writeToA ? _historyTextureA : _historyTextureB;
        ID3D12Resource momentPrev = writeToA ? _momentTextureB : _momentTextureA;
        ID3D12Resource momentOut = writeToA ? _momentTextureA : _momentTextureB;

        Transition(_accumTexture, ref _accumState, ResourceStates.UnorderedAccess);
        Transition(_normalTexture, ref _normalState, ResourceStates.UnorderedAccess);
        Transition(_depthTexture, ref _depthState, ResourceStates.UnorderedAccess);

        _commandList.SetPipelineState(_rayPipelineState);
        int dispatchX = (_width + 7) / 8;
        int dispatchY = (_height + 7) / 8;
        _commandList.Dispatch(dispatchX, dispatchY, 1);

        Transition(_accumTexture, ref _accumState, ResourceStates.NonPixelShaderResource);
        Transition(_normalTexture, ref _normalState, ResourceStates.NonPixelShaderResource);
        Transition(_depthTexture, ref _depthState, ResourceStates.NonPixelShaderResource);

        if (writeToA)
        {
            Transition(_historyTextureB, ref _historyBState, ResourceStates.NonPixelShaderResource);
            Transition(_momentTextureB, ref _momentBState, ResourceStates.NonPixelShaderResource);
            Transition(_historyTextureA, ref _historyAState, ResourceStates.UnorderedAccess);
            Transition(_momentTextureA, ref _momentAState, ResourceStates.UnorderedAccess);
        }
        else
        {
            Transition(_historyTextureA, ref _historyAState, ResourceStates.NonPixelShaderResource);
            Transition(_momentTextureA, ref _momentAState, ResourceStates.NonPixelShaderResource);
            Transition(_historyTextureB, ref _historyBState, ResourceStates.UnorderedAccess);
            Transition(_momentTextureB, ref _momentBState, ResourceStates.UnorderedAccess);
        }

        _commandList.SetPipelineState(_temporalPipelineState);
        _commandList.Dispatch(dispatchX, dispatchY, 1);

        if (writeToA)
        {
            Transition(_historyTextureA, ref _historyAState, ResourceStates.NonPixelShaderResource);
            Transition(_momentTextureA, ref _momentAState, ResourceStates.NonPixelShaderResource);
        }
        else
        {
            Transition(_historyTextureB, ref _historyBState, ResourceStates.NonPixelShaderResource);
            Transition(_momentTextureB, ref _momentBState, ResourceStates.NonPixelShaderResource);
        }

        Transition(_computeTexture, ref _computeState, ResourceStates.UnorderedAccess);
        _commandList.SetPipelineState(_atrousPipelineState);
        _commandList.Dispatch(dispatchX, dispatchY, 1);

        Transition(_computeTexture, ref _computeState, ResourceStates.CopySource);

        _commandList.CopyTextureRegion(
            new TextureCopyLocation(backBuffer, 0),
            0,
            0,
            0,
            new TextureCopyLocation(_computeTexture, 0),
            null);

        Transition(_computeTexture, ref _computeState, ResourceStates.UnorderedAccess);

        _commandList.ResourceBarrier(ResourceBarrier.BarrierTransition(
            backBuffer,
            ResourceStates.CopyDestination,
            ResourceStates.Present));

        _commandList.Close();
        _commandQueue.ExecuteCommandLists(new[] { _commandList });
        _swapChain.Present(_vSync ? 1 : 0, PresentFlags.None);

        if (_accumulationFrame < int.MaxValue)
        {
            _accumulationFrame++;
        }

        SignalFrame();
        _frameIndex = _swapChain.CurrentBackBufferIndex;
        WaitForFrame(_frameIndex);
    }

    public void UpdateCamera(float deltaTime, Vector3 moveInput, Vector2 lookDelta)
    {
        const float mouseSensitivity = 0.003f;
        const float moveSpeed = 3.0f;

        bool cameraDirty = lookDelta.LengthSquared() > 0.0f || moveInput.LengthSquared() > 0.0f;
        _cameraYaw += lookDelta.X * mouseSensitivity;
        _cameraPitch += lookDelta.Y * mouseSensitivity;
        _cameraPitch = Math.Clamp(_cameraPitch, -1.5f, 1.5f);

        UpdateCameraBasis();

        var move = _cameraRight * moveInput.X + _cameraUp * moveInput.Y + _cameraForward * moveInput.Z;
        if (move.LengthSquared() > 0.0001f)
        {
            move = Vector3.Normalize(move);
            _cameraPosition += move * moveSpeed * deltaTime;
        }

        if (cameraDirty)
        {
            _accumulationFrame = 0;
        }
    }

    public void UpdateSettings(int samplesPerPixel, int maxBounces, float ambientStrength, bool vSync)
    {
        samplesPerPixel = Math.Clamp(samplesPerPixel, 1, 8);
        maxBounces = Math.Clamp(maxBounces, 1, 8);
        ambientStrength = Math.Clamp(ambientStrength, 0.0f, 0.2f);

        if (_samplesPerPixel != samplesPerPixel ||
            _maxBounces != maxBounces ||
            Math.Abs(_ambientStrength - ambientStrength) > 0.0001f)
        {
            _accumulationFrame = 0;
        }

        _samplesPerPixel = samplesPerPixel;
        _maxBounces = maxBounces;
        _ambientStrength = ambientStrength;
        _vSync = vSync;
    }

    public void Resize(int width, int height)
    {
        width = Math.Max(width, 1);
        height = Math.Max(height, 1);
        if (width == _width && height == _height)
        {
            return;
        }

        WaitForGpu();
        for (int i = 0; i < FrameCount; i++)
        {
            _backBuffers[i]?.Dispose();
            _backBuffers[i] = null!;
        }

        _swapChain.ResizeBuffers(FrameCount, width, height, Format.R8G8B8A8_UNorm, SwapChainFlags.None);
        _frameIndex = _swapChain.CurrentBackBufferIndex;
        _width = width;
        _height = height;

        CreateBackBuffers();
        CreateComputeTexture();
        _accumulationFrame = 0;
    }

    public void Dispose()
    {
        WaitForGpu();

        for (int i = 0; i < FrameCount; i++)
        {
            _backBuffers[i]?.Dispose();
        }

        _computeTexture.Dispose();
        _accumTexture.Dispose();
        _normalTexture.Dispose();
        _depthTexture.Dispose();
        _historyTextureA.Dispose();
        _historyTextureB.Dispose();
        _momentTextureA.Dispose();
        _momentTextureB.Dispose();
        _constantBuffer.Dispose();
        _sphereBuffer.Dispose();
        _planeBuffer.Dispose();
        _boxBuffer.Dispose();
        _materialBuffer.Dispose();
        _lightBuffer.Dispose();
        _gridCellStartBuffer.Dispose();
        _gridCellCountBuffer.Dispose();
        _gridIndexBuffer.Dispose();
        _uavHeap.Dispose();
        _rayPipelineState.Dispose();
        _temporalPipelineState.Dispose();
        _atrousPipelineState.Dispose();
        _computeRootSignature.Dispose();
        _commandList.Dispose();
        _commandAllocator.Dispose();
        _commandQueue.Dispose();
        _fence.Dispose();
        _fenceEvent.Dispose();
        _swapChain.Dispose();
        _device.Dispose();
    }

    private static IDXGIAdapter1 PickAdapter(IDXGIFactory4 factory)
    {
        for (int i = 0; factory.EnumAdapters1(i, out var adapter).Success; i++)
        {
            var desc = adapter.Description1;
            if ((desc.Flags & AdapterFlags.Software) != 0)
            {
                adapter.Dispose();
                continue;
            }
            return adapter;
        }

        factory.EnumWarpAdapter<IDXGIAdapter1>(out var warpAdapter);
        return warpAdapter ?? throw new InvalidOperationException("No DXGI adapter available.");
    }

    private void CreateBackBuffers()
    {
        for (int i = 0; i < FrameCount; i++)
        {
            _backBuffers[i] = _swapChain.GetBuffer<ID3D12Resource>(i);
        }
    }

    private void CreateComputeResources()
    {
        _computeRootSignature = CreateComputeRootSignature();
        _rayPipelineState = CreateComputePipelineState(_computeRootSignature, "RayGen");
        _temporalPipelineState = CreateComputePipelineState(_computeRootSignature, "TemporalAccumulation");
        _atrousPipelineState = CreateComputePipelineState(_computeRootSignature, "AtrousFilter");
        _uavHeap = _device.CreateDescriptorHeap(new DescriptorHeapDescription(
            DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
            DescriptorCount,
            DescriptorHeapFlags.ShaderVisible));
        _cbvUavDescriptorSize = _device.GetDescriptorHandleIncrementSize(
            DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView);

        CreateConstantBuffer();
        CreateSceneBuffers();
        CreateComputeTexture();
    }

    private ID3D12RootSignature CreateComputeRootSignature()
    {
        var ranges = new[]
        {
            new DescriptorRange1(DescriptorRangeType.ConstantBufferView, 1, 0),
            new DescriptorRange1(DescriptorRangeType.ShaderResourceView, SrvBufferCount + SrvTextureCount + SrvGridCount, 0),
            new DescriptorRange1(DescriptorRangeType.UnorderedAccessView, 8, 0)
        };

        var rootParams = new[]
        {
            new RootParameter1(new RootDescriptorTable1(ranges), ShaderVisibility.All)
        };

        var desc = new RootSignatureDescription1(
            RootSignatureFlags.None,
            rootParams,
            Array.Empty<StaticSamplerDescription>());

        return _device.CreateRootSignature(desc);
    }

    private ID3D12PipelineState CreateComputePipelineState(ID3D12RootSignature rootSignature, string entryPoint)
    {
        const string shaderSource = @"
cbuffer Params : register(b0)
{
    float Time;
    int SphereCount;
    int PlaneCount;
    int BoxCount;
    int LightCount;
    float3 CameraPos;
    float CameraFov;
    float3 CameraForward;
    float _pad0;
    float3 CameraRight;
    float _pad1;
    float3 CameraUp;
    float _pad2;
    int FrameIndex;
    int SamplesPerPixel;
    int MaxBounces;
    float AmbientStrength;
    int GridDimX;
    int GridDimY;
    int GridDimZ;
    float _pad4;
    float3 GridMin;
    float _pad5;
    float3 GridMax;
    float _pad6;
};

struct Sphere
{
    float3 Center;
    float Radius;
    int MaterialId;
    float3 _pad;
};

struct Plane
{
    float3 Normal;
    float D;
    int MaterialId;
    float3 _pad;
};

struct Box
{
    float3 Min;
    float _pad0;
    float3 Max;
    float _pad1;
    int MaterialId;
    float3 _pad2;
};

struct Material
{
    float3 Albedo;
    float Roughness;
    float Metallic;
    float Emission;
    float2 _pad;
};

struct Light
{
    float3 Position;
    float Intensity;
    float3 Color;
    float _pad;
};

StructuredBuffer<Sphere> Spheres : register(t0);
StructuredBuffer<Plane> Planes : register(t1);
StructuredBuffer<Box> Boxes : register(t2);
StructuredBuffer<Material> Materials : register(t3);
StructuredBuffer<Light> Lights : register(t4);
Texture2D<float4> CurrentColor : register(t5);
Texture2D<float4> NormalTex : register(t6);
Texture2D<float> DepthTex : register(t7);
Texture2D<float4> HistoryA : register(t8);
Texture2D<float4> HistoryB : register(t9);
Texture2D<float2> MomentA : register(t10);
Texture2D<float2> MomentB : register(t11);
StructuredBuffer<int> GridCellStart : register(t12);
StructuredBuffer<int> GridCellCount : register(t13);
StructuredBuffer<int> GridIndices : register(t14);

RWTexture2D<float4> Output : register(u0);
RWTexture2D<float4> Accumulation : register(u1);
RWTexture2D<float4> NormalOut : register(u2);
RWTexture2D<float> DepthOut : register(u3);
RWTexture2D<float4> HistoryOutA : register(u4);
RWTexture2D<float4> HistoryOutB : register(u5);
RWTexture2D<float2> MomentOutA : register(u6);
RWTexture2D<float2> MomentOutB : register(u7);

struct HitInfo
{
    float t;
    float3 normal;
    int materialId;
};

HitInfo IntersectScene(float3 rayOrigin, float3 rayDir)
{
    HitInfo hit;
    hit.t = 1e30f;
    hit.normal = float3(0, 0, 0);
    hit.materialId = -1;

    float3 invDir = 1.0f / rayDir;
    float3 t0 = (GridMin - rayOrigin) * invDir;
    float3 t1 = (GridMax - rayOrigin) * invDir;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float tEnter = max(max(tmin.x, tmin.y), tmin.z);
    float tExit = min(min(tmax.x, tmax.y), tmax.z);

    if (tExit > max(tEnter, 0.0f))
    {
        float t = max(tEnter, 0.0f);
        float3 gridSize = GridMax - GridMin;
        float3 cellSize = gridSize / float3(GridDimX, GridDimY, GridDimZ);
        float3 p = rayOrigin + rayDir * t;
        int3 cell = int3(clamp((p - GridMin) / cellSize, 0.0f, float3(GridDimX - 1, GridDimY - 1, GridDimZ - 1)));

        int3 step = int3(rayDir.x > 0 ? 1 : -1, rayDir.y > 0 ? 1 : -1, rayDir.z > 0 ? 1 : -1);
        float3 cellMin = GridMin + float3(cell) * cellSize;
        float3 cellMax = cellMin + cellSize;
        float3 nextBoundary = float3(
            step.x > 0 ? cellMax.x : cellMin.x,
            step.y > 0 ? cellMax.y : cellMin.y,
            step.z > 0 ? cellMax.z : cellMin.z);
        float3 tMax = (nextBoundary - rayOrigin) * invDir;
        float3 tDelta = abs(cellSize * invDir);

        [loop]
        for (int iter = 0; iter < 256; iter++)
        {
            if (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
                cell.x >= GridDimX || cell.y >= GridDimY || cell.z >= GridDimZ)
            {
                break;
            }

            int cellIndex = cell.x + cell.y * GridDimX + cell.z * GridDimX * GridDimY;
            int start = GridCellStart[cellIndex];
            int count = GridCellCount[cellIndex];
            for (int i = 0; i < count; i++)
            {
                int sphereIndex = GridIndices[start + i];
                Sphere s = Spheres[sphereIndex];
                float3 oc = rayOrigin - s.Center;
                float b = dot(oc, rayDir);
                float c = dot(oc, oc) - s.Radius * s.Radius;
                float h = b * b - c;
                if (h > 0.0f)
                {
                    float tHit = -b - sqrt(h);
                    if (tHit > 0.001f && tHit < hit.t)
                    {
                        hit.t = tHit;
                        float3 hitPos = rayOrigin + rayDir * tHit;
                        hit.normal = normalize(hitPos - s.Center);
                        hit.materialId = s.MaterialId;
                    }
                }
            }

            if (tMax.x < tMax.y)
            {
                if (tMax.x < tMax.z)
                {
                    t = tMax.x;
                    tMax.x += tDelta.x;
                    cell.x += step.x;
                }
                else
                {
                    t = tMax.z;
                    tMax.z += tDelta.z;
                    cell.z += step.z;
                }
            }
            else
            {
                if (tMax.y < tMax.z)
                {
                    t = tMax.y;
                    tMax.y += tDelta.y;
                    cell.y += step.y;
                }
                else
                {
                    t = tMax.z;
                    tMax.z += tDelta.z;
                    cell.z += step.z;
                }
            }

            if (t > tExit || t > hit.t)
            {
                break;
            }
        }
    }

    for (int i = 0; i < PlaneCount; i++)
    {
        Plane p = Planes[i];
        float denom = dot(p.Normal, rayDir);
        if (abs(denom) > 1e-5f)
        {
            float tHit = -(dot(p.Normal, rayOrigin) + p.D) / denom;
            if (tHit > 0.001f && tHit < hit.t)
            {
                hit.t = tHit;
                hit.normal = normalize(p.Normal);
                hit.materialId = p.MaterialId;
            }
        }
    }

    for (int i = 0; i < BoxCount; i++)
    {
        Box b = Boxes[i];
        float3 invDir = 1.0f / rayDir;
        float3 t0 = (b.Min - rayOrigin) * invDir;
        float3 t1 = (b.Max - rayOrigin) * invDir;
        float3 tmin = min(t0, t1);
        float3 tmax = max(t0, t1);
        float tNear = max(max(tmin.x, tmin.y), tmin.z);
        float tFar = min(min(tmax.x, tmax.y), tmax.z);
        if (tNear <= tFar && tFar > 0.001f && tNear < hit.t)
        {
            float tHit = tNear > 0.001f ? tNear : tFar;
            if (tHit < hit.t)
            {
                hit.t = tHit;
                float3 hitPos = rayOrigin + rayDir * tHit;
                float3 center = (b.Min + b.Max) * 0.5f;
                float3 local = hitPos - center;
                float3 extents = (b.Max - b.Min) * 0.5f;
                float3 n = float3(0, 0, 0);
                float3 d = abs(local) - extents;
                if (abs(d.x) > abs(d.y) && abs(d.x) > abs(d.z)) n = float3(sign(local.x), 0, 0);
                else if (abs(d.y) > abs(d.z)) n = float3(0, sign(local.y), 0);
                else n = float3(0, 0, sign(local.z));
                hit.normal = n;
                hit.materialId = b.MaterialId;
            }
        }
    }

    return hit;
}

float3 Shade(float3 hitPos, float3 normal, Material mat)
{
    float3 ambient = float3(AmbientStrength, AmbientStrength, AmbientStrength + 0.01f);
    float3 lighting = float3(0, 0, 0);
    for (int i = 0; i < LightCount; i++)
    {
        Light light = Lights[i];
        float3 toLight = light.Position - hitPos;
        float dist2 = dot(toLight, toLight);
        float3 ldir = normalize(toLight);
        float ndotl = max(0.0f, dot(normal, ldir));
        lighting += light.Color * (light.Intensity * ndotl / max(dist2, 0.001f));
    }

    return mat.Albedo * (lighting + ambient) + mat.Emission;
}

uint Hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

float RngNext(inout uint state)
{
    state = Hash(state);
    return (state & 0x00FFFFFF) / 16777216.0f;
}

float3 SampleHemisphere(float3 n, inout uint state)
{
    float u1 = RngNext(state);
    float u2 = RngNext(state);
    float r = sqrt(u1);
    float phi = 6.2831853f * u2;

    float3 tangent = normalize(abs(n.y) < 0.999f ? cross(n, float3(0, 1, 0)) : cross(n, float3(1, 0, 0)));
    float3 bitangent = cross(n, tangent);
    float3 dir = tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + n * sqrt(1.0f - u1);
    return normalize(dir);
}

bool WriteHistoryToA()
{
    return (FrameIndex & 1) == 0;
}

float4 LoadHistoryPrev(int2 pos)
{
    return WriteHistoryToA() ? HistoryB[pos] : HistoryA[pos];
}

float2 LoadMomentPrev(int2 pos)
{
    return WriteHistoryToA() ? MomentB[pos] : MomentA[pos];
}

void StoreHistoryOut(int2 pos, float4 color)
{
    if (WriteHistoryToA())
    {
        HistoryOutA[pos] = color;
    }
    else
    {
        HistoryOutB[pos] = color;
    }
}

void StoreMomentOut(int2 pos, float2 moment)
{
    if (WriteHistoryToA())
    {
        MomentOutA[pos] = moment;
    }
    else
    {
        MomentOutB[pos] = moment;
    }
}

float4 LoadHistoryOut(int2 pos)
{
    return WriteHistoryToA() ? HistoryOutA[pos] : HistoryOutB[pos];
}

[numthreads(8, 8, 1)]
void RayGen(uint3 id : SV_DispatchThreadID)
{
    uint width;
    uint height;
    Output.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
    {
        return;
    }

    uint rngState = Hash(id.x * 1973u + id.y * 9277u + (uint)FrameIndex * 26699u);
    float2 jitter = float2(RngNext(rngState), RngNext(rngState)) - 0.5f;
    float2 uv = (float2(id.xy) + 0.5f + jitter) / float2(width, height);
    uv = uv * 2.0f - 1.0f;
    uv.x *= (float)width / (float)height;

    float tanHalfFov = tan(radians(CameraFov * 0.5f));
    float3 rayOrigin = CameraPos;
    float3 rayDir = normalize(CameraForward +
                              (uv.x * tanHalfFov) * CameraRight +
                              (-uv.y * tanHalfFov) * CameraUp);

    float3 colorSum = float3(0.0f, 0.0f, 0.0f);
    float hitDepth = 1e30f;
    float3 hitNormal = float3(0, 0, 0);
    int spp = max(1, SamplesPerPixel);
    int maxBounces = max(1, MaxBounces);

    for (int s = 0; s < spp; s++)
    {
        float2 sampleJitter = float2(RngNext(rngState), RngNext(rngState)) - 0.5f;
        float2 suv = (float2(id.xy) + 0.5f + sampleJitter) / float2(width, height);
        suv = suv * 2.0f - 1.0f;
        suv.x *= (float)width / (float)height;

        float3 sOrigin = CameraPos;
        float3 sDir = normalize(CameraForward +
                                (suv.x * tanHalfFov) * CameraRight +
                                (-suv.y * tanHalfFov) * CameraUp);

        float3 color = float3(0.02f, 0.02f, 0.03f);
        float3 throughput = float3(1.0f, 1.0f, 1.0f);

        for (int bounce = 0; bounce < maxBounces; bounce++)
        {
            HitInfo hit = IntersectScene(sOrigin, sDir);
            if (hit.materialId < 0)
            {
                color += throughput * float3(0.02f, 0.02f, 0.03f);
                break;
            }

            Material mat = Materials[hit.materialId];
            float3 hitPos = sOrigin + sDir * hit.t;
            float3 normal = hit.normal;
            if (bounce == 0 && s == 0)
            {
                hitDepth = hit.t;
                hitNormal = normal;
            }

            color += throughput * Shade(hitPos, normal, mat);

            float reflectivity = lerp(0.04f, 1.0f, mat.Metallic);
            throughput *= mat.Albedo * reflectivity;
            sOrigin = hitPos + normal * 0.001f;
            float3 perfect = reflect(sDir, normal);
            float rough = saturate(mat.Roughness);
            float3 hemi = SampleHemisphere(perfect, rngState);
            sDir = normalize(lerp(perfect, hemi, rough * rough));
        }

        colorSum += color;
    }

    float3 averaged = colorSum / (float)spp;
    Accumulation[id.xy] = float4(averaged, 1.0f);
    NormalOut[id.xy] = float4(hitNormal, 1.0f);
    DepthOut[id.xy] = hitDepth;
}

[numthreads(8, 8, 1)]
void TemporalAccumulation(uint3 id : SV_DispatchThreadID)
{
    uint width;
    uint height;
    Output.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
    {
        return;
    }

    int2 pos = int2(id.xy);
    float3 current = CurrentColor[pos].xyz;
    float lum = dot(current, float3(0.299f, 0.587f, 0.114f));

    if (FrameIndex == 0)
    {
        StoreHistoryOut(pos, float4(current, 1.0f));
        StoreMomentOut(pos, float2(lum, lum * lum));
        return;
    }

    float3 history = LoadHistoryPrev(pos).xyz;
    float2 moment = LoadMomentPrev(pos);

    float alpha = 0.1f;
    float3 blended = lerp(history, current, alpha);
    float2 blendedMoment = lerp(moment, float2(lum, lum * lum), alpha);

    StoreHistoryOut(pos, float4(blended, 1.0f));
    StoreMomentOut(pos, blendedMoment);
}

[numthreads(8, 8, 1)]
void AtrousFilter(uint3 id : SV_DispatchThreadID)
{
    uint width;
    uint height;
    Output.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
    {
        return;
    }

    int2 pos = int2(id.xy);
    float3 centerColor = LoadHistoryOut(pos).xyz;
    float3 centerNormal = NormalTex[pos].xyz;
    float centerDepth = DepthTex[pos];

    float3 sum = 0.0f;
    float wsum = 0.0f;
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            int2 p = pos + int2(x, y);
            if (p.x < 0 || p.y < 0 || p.x >= (int)width || p.y >= (int)height)
            {
                continue;
            }

            float3 c = LoadHistoryOut(p).xyz;
            float3 n = NormalTex[p].xyz;
            float d = DepthTex[p];

            float ndot = max(0.0f, dot(centerNormal, n));
            float depthWeight = exp(-abs(centerDepth - d) * 2.0f);
            float w = ndot * depthWeight;
            sum += c * w;
            wsum += w;
        }
    }

    float3 filtered = (wsum > 0.0f) ? sum / wsum : centerColor;
    Output[pos] = float4(filtered, 1.0f);
}
";

        byte[] cs = CompileShader(shaderSource, entryPoint, "cs_6_0");
        var psoDesc = new ComputePipelineStateDescription
        {
            RootSignature = rootSignature,
            ComputeShader = new ShaderBytecode(cs)
        };

        return _device.CreateComputePipelineState<ID3D12PipelineState>(psoDesc);
    }

    private void CreateConstantBuffer()
    {
        _constantBuffer?.Dispose();
        _constantBufferSize = AlignTo256(Marshal.SizeOf<ComputeParams>());
        _constantBuffer = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Upload),
            HeapFlags.None,
            ResourceDescription.Buffer((ulong)_constantBufferSize),
            ResourceStates.GenericRead);

        var cbvDesc = new ConstantBufferViewDescription
        {
            BufferLocation = _constantBuffer.GPUVirtualAddress,
            SizeInBytes = _constantBufferSize
        };

        var cbvHandle = GetCpuHandle(CbvIndex);
        _device.CreateConstantBufferView(cbvDesc, cbvHandle);
    }

    private void CreateSceneBuffers()
    {
        const int sphereCount = 10_000;
        _gridDimX = 20;
        _gridDimY = 10;
        _gridDimZ = 20;
        _gridMin = new Vector3(-10.0f, -2.0f, 0.0f);
        _gridMax = new Vector3(10.0f, 6.0f, 20.0f);

        var spheres = new Sphere[sphereCount];
        var rng = new Random(1234);
        Vector3 gridSize = _gridMax - _gridMin;
        for (int i = 0; i < sphereCount; i++)
        {
            float radius = 0.15f + (float)rng.NextDouble() * 0.35f;
            var center = new Vector3(
                _gridMin.X + radius + (float)rng.NextDouble() * (gridSize.X - radius * 2),
                _gridMin.Y + radius + (float)rng.NextDouble() * (gridSize.Y - radius * 2),
                _gridMin.Z + radius + (float)rng.NextDouble() * (gridSize.Z - radius * 2));

            int materialId = rng.Next(0, 6);
            spheres[i] = new Sphere(center, radius, materialId);
        }

        var planes = new[]
        {
            new Plane(Vector3.UnitY, 1.5f, 1),
            new Plane(Vector3.UnitZ, 10.0f, 2),
            new Plane(Vector3.UnitY, 1.0f, 5)
        };

        var boxes = new[]
        {
            new Box(new Vector3(-1.5f, -0.5f, 4.0f), new Vector3(-0.5f, 0.5f, 5.0f), 2),
            new Box(new Vector3(0.8f, -1.0f, 2.8f), new Vector3(1.6f, -0.2f, 3.6f), 5),
            new Box(new Vector3(-3.0f, -1.0f, 6.0f), new Vector3(-2.2f, 0.2f, 6.8f), 6)
        };

        var materials = new[]
        {
            new Material(new Vector3(0.7f, 0.2f, 0.2f), 0.15f, 0.0f, 0.0f),
            new Material(new Vector3(0.2f, 0.7f, 0.2f), 0.6f, 0.0f, 0.0f),
            new Material(new Vector3(0.2f, 0.2f, 0.7f), 0.05f, 0.0f, 0.0f),
            new Material(new Vector3(0.9f, 0.9f, 0.9f), 0.3f, 1.0f, 0.0f),
            new Material(new Vector3(0.9f, 0.7f, 0.2f), 0.7f, 0.6f, 0.0f),
            new Material(new Vector3(0.95f, 0.95f, 0.95f), 0.1f, 1.0f, 0.0f)
        };

        var lights = new[]
        {
            new Light(new Vector3(2.0f, 3.0f, -1.0f), new Vector3(1.0f, 1.0f, 1.0f), 8.0f),
            new Light(new Vector3(-3.0f, 2.5f, 2.0f), new Vector3(0.6f, 0.8f, 1.0f), 6.0f)
        };

        _sphereCount = spheres.Length;
        _planeCount = planes.Length;
        _boxCount = boxes.Length;
        _lightCount = lights.Length;

        ID3D12Resource sphereUpload = null!;
        ID3D12Resource planeUpload = null!;
        ID3D12Resource boxUpload = null!;
        ID3D12Resource materialUpload = null!;
        ID3D12Resource lightUpload = null!;
        ID3D12Resource gridStartUpload = null!;
        ID3D12Resource gridCountUpload = null!;
        ID3D12Resource gridIndexUpload = null!;

        _sphereBuffer = CreateDefaultBuffer(spheres, out sphereUpload);
        _planeBuffer = CreateDefaultBuffer(planes, out planeUpload);
        _boxBuffer = CreateDefaultBuffer(boxes, out boxUpload);
        _materialBuffer = CreateDefaultBuffer(materials, out materialUpload);
        _lightBuffer = CreateDefaultBuffer(lights, out lightUpload);
        BuildGrid(spheres, out int[] cellStart, out int[] cellCount, out int[] indices);
        _gridIndexCount = indices.Length;
        _gridCellStartBuffer = CreateDefaultBuffer(cellStart, out gridStartUpload);
        _gridCellCountBuffer = CreateDefaultBuffer(cellCount, out gridCountUpload);
        _gridIndexBuffer = CreateDefaultBuffer(indices, out gridIndexUpload);

        _commandAllocator.Reset();
        _commandList.Reset(_commandAllocator, null);

        CopyBufferToDefault(_sphereBuffer, sphereUpload);
        CopyBufferToDefault(_planeBuffer, planeUpload);
        CopyBufferToDefault(_boxBuffer, boxUpload);
        CopyBufferToDefault(_materialBuffer, materialUpload);
        CopyBufferToDefault(_lightBuffer, lightUpload);
        CopyBufferToDefault(_gridCellStartBuffer, gridStartUpload);
        CopyBufferToDefault(_gridCellCountBuffer, gridCountUpload);
        CopyBufferToDefault(_gridIndexBuffer, gridIndexUpload);

        TransitionToShaderResource(_sphereBuffer);
        TransitionToShaderResource(_planeBuffer);
        TransitionToShaderResource(_boxBuffer);
        TransitionToShaderResource(_materialBuffer);
        TransitionToShaderResource(_lightBuffer);
        TransitionToShaderResource(_gridCellStartBuffer);
        TransitionToShaderResource(_gridCellCountBuffer);
        TransitionToShaderResource(_gridIndexBuffer);

        _commandList.Close();
        _commandQueue.ExecuteCommandLists(new[] { _commandList });
        WaitForGpu();

        sphereUpload.Dispose();
        planeUpload.Dispose();
        boxUpload.Dispose();
        materialUpload.Dispose();
        lightUpload.Dispose();
        gridStartUpload.Dispose();
        gridCountUpload.Dispose();
        gridIndexUpload.Dispose();

        var srvHandle = GetCpuHandle(SrvBaseIndex);
        CreateStructuredBufferSrv(_sphereBuffer, _sphereCount, Marshal.SizeOf<Sphere>(), srvHandle);

        srvHandle.Ptr += _cbvUavDescriptorSize;
        CreateStructuredBufferSrv(_planeBuffer, _planeCount, Marshal.SizeOf<Plane>(), srvHandle);

        srvHandle.Ptr += _cbvUavDescriptorSize;
        CreateStructuredBufferSrv(_boxBuffer, _boxCount, Marshal.SizeOf<Box>(), srvHandle);

        srvHandle.Ptr += _cbvUavDescriptorSize;
        CreateStructuredBufferSrv(_materialBuffer, materials.Length, Marshal.SizeOf<Material>(), srvHandle);

        srvHandle.Ptr += _cbvUavDescriptorSize;
        CreateStructuredBufferSrv(_lightBuffer, _lightCount, Marshal.SizeOf<Light>(), srvHandle);

        int gridSrvIndex = SrvBaseIndex + SrvBufferCount + SrvTextureCount;
        CreateStructuredBufferSrv(_gridCellStartBuffer, _gridDimX * _gridDimY * _gridDimZ, sizeof(int), GetCpuHandle(gridSrvIndex++));
        CreateStructuredBufferSrv(_gridCellCountBuffer, _gridDimX * _gridDimY * _gridDimZ, sizeof(int), GetCpuHandle(gridSrvIndex++));
        CreateStructuredBufferSrv(_gridIndexBuffer, _gridIndexCount, sizeof(int), GetCpuHandle(gridSrvIndex++));
    }

    private void CreateComputeTexture()
    {
        _computeTexture?.Dispose();
        _accumTexture?.Dispose();
        _normalTexture?.Dispose();
        _depthTexture?.Dispose();
        _historyTextureA?.Dispose();
        _historyTextureB?.Dispose();
        _momentTextureA?.Dispose();
        _momentTextureB?.Dispose();

        var outputDesc = ResourceDescription.Texture2D(
            Format.R8G8B8A8_UNorm,
            (ulong)_width,
            _height,
            1,
            1,
            1,
            0,
            ResourceFlags.AllowUnorderedAccess);

        _computeTexture = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Default),
            HeapFlags.None,
            outputDesc,
            ResourceStates.UnorderedAccess);

        var colorDesc = ResourceDescription.Texture2D(
            Format.R16G16B16A16_Float,
            (ulong)_width,
            _height,
            1,
            1,
            1,
            0,
            ResourceFlags.AllowUnorderedAccess);

        _accumTexture = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Default),
            HeapFlags.None,
            colorDesc,
            ResourceStates.UnorderedAccess);

        _normalTexture = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Default),
            HeapFlags.None,
            colorDesc,
            ResourceStates.UnorderedAccess);

        var depthDesc = ResourceDescription.Texture2D(
            Format.R32_Float,
            (ulong)_width,
            _height,
            1,
            1,
            1,
            0,
            ResourceFlags.AllowUnorderedAccess);

        _depthTexture = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Default),
            HeapFlags.None,
            depthDesc,
            ResourceStates.UnorderedAccess);

        _historyTextureA = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Default),
            HeapFlags.None,
            colorDesc,
            ResourceStates.UnorderedAccess);

        _historyTextureB = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Default),
            HeapFlags.None,
            colorDesc,
            ResourceStates.UnorderedAccess);

        var momentDesc = ResourceDescription.Texture2D(
            Format.R16G16_Float,
            (ulong)_width,
            _height,
            1,
            1,
            1,
            0,
            ResourceFlags.AllowUnorderedAccess);

        _momentTextureA = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Default),
            HeapFlags.None,
            momentDesc,
            ResourceStates.UnorderedAccess);

        _momentTextureB = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Default),
            HeapFlags.None,
            momentDesc,
            ResourceStates.UnorderedAccess);

        var outputUavDesc = new UnorderedAccessViewDescription
        {
            Format = Format.R8G8B8A8_UNorm,
            ViewDimension = UnorderedAccessViewDimension.Texture2D,
            Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
        };

        var colorUavDesc = new UnorderedAccessViewDescription
        {
            Format = Format.R16G16B16A16_Float,
            ViewDimension = UnorderedAccessViewDimension.Texture2D,
            Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
        };

        var depthUavDesc = new UnorderedAccessViewDescription
        {
            Format = Format.R32_Float,
            ViewDimension = UnorderedAccessViewDimension.Texture2D,
            Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
        };

        var momentUavDesc = new UnorderedAccessViewDescription
        {
            Format = Format.R16G16_Float,
            ViewDimension = UnorderedAccessViewDimension.Texture2D,
            Texture2D = new Texture2DUnorderedAccessView { MipSlice = 0 }
        };

        _device.CreateUnorderedAccessView(_computeTexture, null, outputUavDesc, GetCpuHandle(UavBaseIndex + 0));
        _device.CreateUnorderedAccessView(_accumTexture, null, colorUavDesc, GetCpuHandle(UavBaseIndex + 1));
        _device.CreateUnorderedAccessView(_normalTexture, null, colorUavDesc, GetCpuHandle(UavBaseIndex + 2));
        _device.CreateUnorderedAccessView(_depthTexture, null, depthUavDesc, GetCpuHandle(UavBaseIndex + 3));
        _device.CreateUnorderedAccessView(_historyTextureA, null, colorUavDesc, GetCpuHandle(UavBaseIndex + 4));
        _device.CreateUnorderedAccessView(_historyTextureB, null, colorUavDesc, GetCpuHandle(UavBaseIndex + 5));
        _device.CreateUnorderedAccessView(_momentTextureA, null, momentUavDesc, GetCpuHandle(UavBaseIndex + 6));
        _device.CreateUnorderedAccessView(_momentTextureB, null, momentUavDesc, GetCpuHandle(UavBaseIndex + 7));

        var colorSrvDesc = new ShaderResourceViewDescription
        {
            ViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension.Texture2D,
            Shader4ComponentMapping = 0x1688,
            Format = Format.R16G16B16A16_Float,
            Texture2D = new Texture2DShaderResourceView { MipLevels = 1, MostDetailedMip = 0 }
        };

        var depthSrvDesc = new ShaderResourceViewDescription
        {
            ViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension.Texture2D,
            Shader4ComponentMapping = 0x1688,
            Format = Format.R32_Float,
            Texture2D = new Texture2DShaderResourceView { MipLevels = 1, MostDetailedMip = 0 }
        };

        var momentSrvDesc = new ShaderResourceViewDescription
        {
            ViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension.Texture2D,
            Shader4ComponentMapping = 0x1688,
            Format = Format.R16G16_Float,
            Texture2D = new Texture2DShaderResourceView { MipLevels = 1, MostDetailedMip = 0 }
        };

        int srvIndex = SrvBaseIndex + SrvBufferCount;
        _device.CreateShaderResourceView(_accumTexture, colorSrvDesc, GetCpuHandle(srvIndex++));
        _device.CreateShaderResourceView(_normalTexture, colorSrvDesc, GetCpuHandle(srvIndex++));
        _device.CreateShaderResourceView(_depthTexture, depthSrvDesc, GetCpuHandle(srvIndex++));
        _device.CreateShaderResourceView(_historyTextureA, colorSrvDesc, GetCpuHandle(srvIndex++));
        _device.CreateShaderResourceView(_historyTextureB, colorSrvDesc, GetCpuHandle(srvIndex++));
        _device.CreateShaderResourceView(_momentTextureA, momentSrvDesc, GetCpuHandle(srvIndex++));
        _device.CreateShaderResourceView(_momentTextureB, momentSrvDesc, GetCpuHandle(srvIndex++));

        _computeState = ResourceStates.UnorderedAccess;
        _accumState = ResourceStates.UnorderedAccess;
        _normalState = ResourceStates.UnorderedAccess;
        _depthState = ResourceStates.UnorderedAccess;
        _historyAState = ResourceStates.UnorderedAccess;
        _historyBState = ResourceStates.UnorderedAccess;
        _momentAState = ResourceStates.UnorderedAccess;
        _momentBState = ResourceStates.UnorderedAccess;
        _accumulationFrame = 0;
    }

    private static byte[] CompileShader(string source, string entryPoint, string profile)
    {
        IntPtr compilerPtr = IntPtr.Zero;
        IntPtr utilsPtr = IntPtr.Zero;

        Dxc.DxcCreateInstance(Dxc.CLSID_DxcCompiler, typeof(IDxcCompiler).GUID, out compilerPtr);
        Dxc.DxcCreateInstance(Dxc.CLSID_DxcUtils, typeof(IDxcUtils).GUID, out utilsPtr);

        using var compiler = new IDxcCompiler(compilerPtr);
        using var utils = new IDxcUtils(utilsPtr);
        utils.CreateDefaultIncludeHandler(out IDxcIncludeHandler includeHandler);
        if (includeHandler == null)
        {
            throw new InvalidOperationException("Failed to create DXC include handler.");
        }

        byte[] sourceBytes = System.Text.Encoding.UTF8.GetBytes(source);
        var handle = GCHandle.Alloc(sourceBytes, GCHandleType.Pinned);
        try
        {
            utils.CreateBlobFromPinned(handle.AddrOfPinnedObject(), sourceBytes.Length, Dxc.DXC_CP_UTF8, out IDxcBlobEncoding blobEncoding);

            string[] args = { "-Zi", "-Qembed_debug", "-Od" };
            IDxcOperationResult? result = compiler.Compile(blobEncoding, "shader.hlsl", entryPoint, profile, args, Array.Empty<DxcDefine>(), includeHandler);
            if (result == null)
            {
                throw new InvalidOperationException("DXC did not return a compile result.");
            }

            using (result)
            {
            result.GetStatus(out Result status);
            if (status.Failure)
            {
                string errors = GetErrors(result);
                throw new InvalidOperationException(string.IsNullOrWhiteSpace(errors) ? "Shader compilation failed." : errors);
            }

            using IDxcBlob blob = result.GetResult();
            byte[] bytecode = new byte[blob.BufferSize];
            Marshal.Copy(blob.BufferPointer, bytecode, 0, bytecode.Length);
            return bytecode;
            }
        }
        finally
        {
            handle.Free();
        }
    }

    private static string GetErrors(IDxcOperationResult result)
    {
        result.GetErrorBuffer(out IDxcBlobEncoding errorBlob);
        if (errorBlob == null)
        {
            return string.Empty;
        }

        if (errorBlob.BufferSize == 0)
        {
            return string.Empty;
        }

        return Marshal.PtrToStringUTF8(errorBlob.BufferPointer, (int)errorBlob.BufferSize) ?? string.Empty;
    }

    private void SignalFrame()
    {
        _fenceValue++;
        _commandQueue.Signal(_fence, _fenceValue);
        _frameFenceValues[_frameIndex] = _fenceValue;
    }

    private void WaitForFrame(int frameIndex)
    {
        ulong value = _frameFenceValues[frameIndex];
        if (value == 0)
        {
            return;
        }

        if (_fence.CompletedValue < value)
        {
            _fence.SetEventOnCompletion(value, _fenceEvent.SafeWaitHandle.DangerousGetHandle());
            _fenceEvent.WaitOne();
        }
    }

    private void WaitForGpu()
    {
        _fenceValue++;
        _commandQueue.Signal(_fence, _fenceValue);
        if (_fence.CompletedValue < _fenceValue)
        {
            _fence.SetEventOnCompletion(_fenceValue, _fenceEvent.SafeWaitHandle.DangerousGetHandle());
            _fenceEvent.WaitOne();
        }
    }

    private void UpdateComputeConstants()
    {
        float time = (float)_clock.Elapsed.TotalSeconds;
        var data = new ComputeParams(
            time,
            _sphereCount,
            _planeCount,
            _boxCount,
            _lightCount,
            _cameraPosition,
            _cameraFov,
            _cameraForward,
            _cameraRight,
            _cameraUp,
            _accumulationFrame,
            _samplesPerPixel,
            _maxBounces,
            _ambientStrength,
            _gridDimX,
            _gridDimY,
            _gridDimZ,
            _gridMin,
            _gridMax);
        var mapped = _constantBuffer.Map<byte>(0, _constantBufferSize);
        MemoryMarshal.Write(mapped, in data);
        _constantBuffer.Unmap(0);
    }

    private static int AlignTo256(int size) => (size + 255) & ~255;

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct ComputeParams
    {
        public readonly float Time;
        public readonly int SphereCount;
        public readonly int PlaneCount;
        public readonly int BoxCount;
        public readonly int LightCount;
        public readonly Vector3 CameraPos;
        public readonly float CameraFov;
        public readonly Vector3 CameraForward;
        private readonly float _pad0;
        public readonly Vector3 CameraRight;
        private readonly float _pad1;
        public readonly Vector3 CameraUp;
        private readonly float _pad2;
        public readonly int FrameIndex;
        public readonly int SamplesPerPixel;
        public readonly int MaxBounces;
        public readonly float AmbientStrength;
        public readonly int GridDimX;
        public readonly int GridDimY;
        public readonly int GridDimZ;
        private readonly float _pad3;
        public readonly Vector3 GridMin;
        private readonly float _pad4;
        public readonly Vector3 GridMax;
        private readonly float _pad5;

        public ComputeParams(
            float time,
            int sphereCount,
            int planeCount,
            int boxCount,
            int lightCount,
            Vector3 cameraPos,
            float cameraFov,
            Vector3 cameraForward,
            Vector3 cameraRight,
            Vector3 cameraUp,
            int frameIndex,
            int samplesPerPixel,
            int maxBounces,
            float ambientStrength,
            int gridDimX,
            int gridDimY,
            int gridDimZ,
            Vector3 gridMin,
            Vector3 gridMax)
        {
            Time = time;
            SphereCount = sphereCount;
            PlaneCount = planeCount;
            BoxCount = boxCount;
            LightCount = lightCount;
            CameraPos = cameraPos;
            CameraFov = cameraFov;
            CameraForward = cameraForward;
            _pad0 = 0f;
            CameraRight = cameraRight;
            _pad1 = 0f;
            CameraUp = cameraUp;
            _pad2 = 0f;
            FrameIndex = frameIndex;
            SamplesPerPixel = samplesPerPixel;
            MaxBounces = maxBounces;
            AmbientStrength = ambientStrength;
            GridDimX = gridDimX;
            GridDimY = gridDimY;
            GridDimZ = gridDimZ;
            _pad3 = 0f;
            GridMin = gridMin;
            _pad4 = 0f;
            GridMax = gridMax;
            _pad5 = 0f;
        }
    }

    private void UpdateCameraBasis()
    {
        float cosPitch = MathF.Cos(_cameraPitch);
        float sinPitch = MathF.Sin(_cameraPitch);
        float cosYaw = MathF.Cos(_cameraYaw);
        float sinYaw = MathF.Sin(_cameraYaw);

        _cameraForward = Vector3.Normalize(new Vector3(sinYaw * cosPitch, sinPitch, cosYaw * cosPitch));
        _cameraRight = Vector3.Normalize(Vector3.Cross(_cameraForward, Vector3.UnitY));
        _cameraUp = Vector3.Normalize(Vector3.Cross(_cameraRight, _cameraForward));
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct Sphere
    {
        public readonly Vector3 Center;
        public readonly float Radius;
        public readonly int MaterialId;
        private readonly Vector3 _pad;

        public Sphere(Vector3 center, float radius, int materialId)
        {
            Center = center;
            Radius = radius;
            MaterialId = materialId;
            _pad = default;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct Plane
    {
        public readonly Vector3 Normal;
        public readonly float D;
        public readonly int MaterialId;
        private readonly Vector3 _pad;

        public Plane(Vector3 normal, float d, int materialId)
        {
            Normal = normal;
            D = d;
            MaterialId = materialId;
            _pad = default;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct Box
    {
        public readonly Vector3 Min;
        private readonly float _pad0;
        public readonly Vector3 Max;
        private readonly float _pad1;
        public readonly int MaterialId;
        private readonly Vector3 _pad2;

        public Box(Vector3 min, Vector3 max, int materialId)
        {
            Min = min;
            _pad0 = 0f;
            Max = max;
            _pad1 = 0f;
            MaterialId = materialId;
            _pad2 = default;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct Material
    {
        public readonly Vector3 Albedo;
        public readonly float Roughness;
        public readonly float Metallic;
        public readonly float Emission;
        private readonly Vector2 _pad;

        public Material(Vector3 albedo, float roughness, float metallic, float emission)
        {
            Albedo = albedo;
            Roughness = roughness;
            Metallic = metallic;
            Emission = emission;
            _pad = default;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct Light
    {
        public readonly Vector3 Position;
        public readonly float Intensity;
        public readonly Vector3 Color;
        private readonly float _pad;

        public Light(Vector3 position, Vector3 color, float intensity)
        {
            Position = position;
            Intensity = intensity;
            Color = color;
            _pad = 0f;
        }
    }

    private ID3D12Resource CreateDefaultBuffer<T>(T[] data, out ID3D12Resource uploadBuffer) where T : struct
    {
        int stride = Marshal.SizeOf<T>();
        int sizeInBytes = stride * data.Length;

        var defaultBuffer = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Default),
            HeapFlags.None,
            ResourceDescription.Buffer((ulong)sizeInBytes),
            ResourceStates.CopyDestination);

        uploadBuffer = _device.CreateCommittedResource(
            new HeapProperties(HeapType.Upload),
            HeapFlags.None,
            ResourceDescription.Buffer((ulong)sizeInBytes),
            ResourceStates.GenericRead);

        var bytes = MemoryMarshal.AsBytes(data.AsSpan());
        var mapped = uploadBuffer.Map<byte>(0, sizeInBytes);
        bytes.CopyTo(mapped);
        uploadBuffer.Unmap(0);

        return defaultBuffer;
    }

    private void CopyBufferToDefault(ID3D12Resource defaultBuffer, ID3D12Resource uploadBuffer)
    {
        _commandList.CopyBufferRegion(defaultBuffer, 0, uploadBuffer, 0, uploadBuffer.Description.Width);
    }

    private void TransitionToShaderResource(ID3D12Resource buffer)
    {
        _commandList.ResourceBarrier(ResourceBarrier.BarrierTransition(
            buffer,
            ResourceStates.CopyDestination,
            ResourceStates.NonPixelShaderResource));
    }

    private void CreateStructuredBufferSrv(ID3D12Resource buffer, int elementCount, int stride, CpuDescriptorHandle handle)
    {
        var srvDesc = new ShaderResourceViewDescription
        {
            ViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension.Buffer,
            Shader4ComponentMapping = 0x1688,
            Format = Format.Unknown,
            Buffer = new BufferShaderResourceView
            {
                FirstElement = 0,
                NumElements = elementCount,
                StructureByteStride = stride,
                Flags = BufferShaderResourceViewFlags.None
            }
        };

        _device.CreateShaderResourceView(buffer, srvDesc, handle);
    }

    private void BuildGrid(Sphere[] spheres, out int[] cellStart, out int[] cellCount, out int[] indices)
    {
        int cellTotal = _gridDimX * _gridDimY * _gridDimZ;
        var cells = new List<int>[cellTotal];
        for (int i = 0; i < cellTotal; i++)
        {
            cells[i] = new List<int>();
        }

        Vector3 gridSize = _gridMax - _gridMin;
        Vector3 cellSize = new Vector3(
            gridSize.X / _gridDimX,
            gridSize.Y / _gridDimY,
            gridSize.Z / _gridDimZ);

        for (int i = 0; i < spheres.Length; i++)
        {
            Sphere s = spheres[i];
            Vector3 min = s.Center - new Vector3(s.Radius);
            Vector3 max = s.Center + new Vector3(s.Radius);

            int minX = Math.Clamp((int)((min.X - _gridMin.X) / cellSize.X), 0, _gridDimX - 1);
            int minY = Math.Clamp((int)((min.Y - _gridMin.Y) / cellSize.Y), 0, _gridDimY - 1);
            int minZ = Math.Clamp((int)((min.Z - _gridMin.Z) / cellSize.Z), 0, _gridDimZ - 1);
            int maxX = Math.Clamp((int)((max.X - _gridMin.X) / cellSize.X), 0, _gridDimX - 1);
            int maxY = Math.Clamp((int)((max.Y - _gridMin.Y) / cellSize.Y), 0, _gridDimY - 1);
            int maxZ = Math.Clamp((int)((max.Z - _gridMin.Z) / cellSize.Z), 0, _gridDimZ - 1);

            for (int z = minZ; z <= maxZ; z++)
            {
                for (int y = minY; y <= maxY; y++)
                {
                    for (int x = minX; x <= maxX; x++)
                    {
                        int cellIndex = x + y * _gridDimX + z * _gridDimX * _gridDimY;
                        cells[cellIndex].Add(i);
                    }
                }
            }
        }

        cellStart = new int[cellTotal];
        cellCount = new int[cellTotal];
        int totalIndices = 0;
        for (int i = 0; i < cellTotal; i++)
        {
            cellStart[i] = totalIndices;
            cellCount[i] = cells[i].Count;
            totalIndices += cells[i].Count;
        }

        indices = new int[totalIndices];
        int offset = 0;
        for (int i = 0; i < cellTotal; i++)
        {
            var list = cells[i];
            list.CopyTo(indices, offset);
            offset += list.Count;
        }
    }

    private void Transition(ID3D12Resource resource, ref ResourceStates current, ResourceStates next)
    {
        if (current == next)
        {
            return;
        }

        _commandList.ResourceBarrier(ResourceBarrier.BarrierTransition(resource, current, next));
        current = next;
    }

    private CpuDescriptorHandle GetCpuHandle(int index)
    {
        var handle = _uavHeap.GetCPUDescriptorHandleForHeapStart();
        handle.Ptr += _cbvUavDescriptorSize * index;
        return handle;
    }
}
