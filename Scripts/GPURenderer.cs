using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using System.Runtime.InteropServices; 

public class GPURenderer : MonoBehaviour
{
    [Header("文件设置")]
    public string csvFileName = "unity_cell_data.csv";

    [Header("渲染设置")]
    public Mesh cellMesh;           
    public Material cellMaterial;   
    public Gradient colorGradient;  

    [Header("参数")]
    public float positionScale = 0.5f;
    public float heightMultiplier = 10.0f;
    public float baseScale = 0.5f;

    public struct CellDataGPU
    {
        public Vector3 position;
        public float scale;
        public Vector4 color;
    }


    private ComputeBuffer cellBuffer;     // 数据 Buffer
    private ComputeBuffer argsBuffer;     // 参数 Buffer (告诉 GPU 画多少个)
    private uint[] args = new uint[5] { 0, 0, 0, 0, 0 };
    private int cellCount = 0;
    private Bounds bounds;                // 渲染边界
    // 保存内存中的数据副本，以便修改
    public List<CellDataGPU> cellDataList = new List<CellDataGPU>();
    void Start()
    {
        List<CellDataGPU> dataList = LoadDataFromCSV();
        cellCount = dataList.Count;

        if (cellCount == 0) return;

        InitializeBuffers(dataList);

        bounds = new Bounds(Vector3.zero, new Vector3(10000, 10000, 10000));
    }

    void Update()
    {
        // 每一帧命令 GPU 渲染
        if (cellCount > 0 && cellMaterial != null && cellMesh != null)
        {
            Graphics.DrawMeshInstancedIndirect(
                cellMesh,
                0,
                cellMaterial,
                bounds,
                argsBuffer
            );
        }
    }

    void InitializeBuffers(List<CellDataGPU> data)
    {

        if (cellBuffer != null) cellBuffer.Release();
        if (argsBuffer != null) argsBuffer.Release();


        // 32 是 stride (步长): Vector3(12) + float(4) + Vector4(16) = 32 bytes
        cellBuffer = new ComputeBuffer(cellCount, 32);
        cellBuffer.SetData(data); // 把数据推送到 GPU


        argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        args[0] = (uint)cellMesh.GetIndexCount(0); // 网格顶点数
        args[1] = (uint)cellCount;                 // 实例数量 (几万个)
        args[2] = (uint)cellMesh.GetIndexStart(0);
        args[3] = (uint)cellMesh.GetBaseVertex(0);
        args[4] = 0;
        argsBuffer.SetData(args);

        cellMaterial.SetBuffer("_CellBuffer", cellBuffer);

        cellMaterial.EnableKeyword("PROCEDURAL_INSTANCING_ON");
    }


    void OnDisable()
    {
        if (cellBuffer != null) cellBuffer.Release();
        if (argsBuffer != null) argsBuffer.Release();
        cellBuffer = null;
        argsBuffer = null;
    }

    List<CellDataGPU> LoadDataFromCSV()
    {
        List<CellDataGPU> list = new List<CellDataGPU>();
        string path = Path.Combine(Application.streamingAssetsPath, csvFileName);

        if (!File.Exists(path)) { Debug.LogError("找不到文件"); return list; }
        else Debug.Log("找到文件");

        string[] lines = File.ReadAllLines(path);
        for (int i = 1; i < lines.Length; i++)
        {
            if (string.IsNullOrEmpty(lines[i])) continue;
            string[] values = lines[i].Split(',');
            if (values.Length < 6) continue;

            try
            {
                float x = float.Parse(values[1], CultureInfo.InvariantCulture);
                float y = float.Parse(values[2], CultureInfo.InvariantCulture);
                float expr = float.Parse(values[4], CultureInfo.InvariantCulture);

                CellDataGPU cell = new CellDataGPU();
                // 坐标转换
                cell.position = new Vector3(x * positionScale, expr * heightMultiplier, y * positionScale);
                // 颜色转换
                Color c = colorGradient.Evaluate(expr);
                cell.color = new Vector4(c.r, c.g, c.b, 1);
                // 大小
                cell.scale = baseScale * (0.8f + expr);

                list.Add(cell);
            }
            catch { }
        }
        Debug.Log($"CSV加载完成，共读取到 {list.Count} 个细胞数据。");
        return list;
    }



    public void InitializeData(List<CellDataGPU> data)
    {
        this.cellDataList = data;
        this.cellCount = data.Count;

        if (cellBuffer != null) cellBuffer.Release();
        cellBuffer = new ComputeBuffer(cellCount, Marshal.SizeOf(typeof(CellDataGPU)));
        cellBuffer.SetData(cellDataList.ToArray());
        cellMaterial.SetBuffer("_CellData", cellBuffer);
    }

    public void UpdateColorsForRegions(List<int> regionIds, Color[] palette)
    {
        if (cellDataList == null || cellDataList.Count == 0) return;

        for (int i = 0; i < cellDataList.Count; i++)
        {
            if (i >= regionIds.Count) break;

            // 根据 Region ID 取色
            int rId = regionIds[i];
            Color c = palette[rId % palette.Length];

            // 更新内存副本
            CellDataGPU temp = cellDataList[i];
            temp.color = new Vector4(c.r, c.g, c.b, 1.0f); // 确保 Alpha 是 1
            cellDataList[i] = temp;
        }

        //将更新后的全量数据重新上传到显卡
        cellBuffer.SetData(cellDataList.ToArray());

        Debug.Log($"[GPU] 已刷新 {cellCount} 个细胞的区域颜色");
    }
}