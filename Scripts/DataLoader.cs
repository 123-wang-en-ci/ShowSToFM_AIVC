using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Globalization;
using UnityEngine.Networking;
using System.Xml.Serialization;
using UnityEngine.UI;
using TMPro;

public class DataLoader : MonoBehaviour
{
    [Header("设置")]
    public string csvFileName = "unity_cell_data.csv";
    public GameObject cellPrefab;

    [Header("显示参数")]
    public float positionScale = 0.5f;
    public float heightMultiplier = 1.0f;
    public float CellScale = 5.0f;

    [Header("视觉增强设置")]
    public Gradient colorGradient;
    public float emissionIntensity = 2.0f;

    public Dictionary<string, GameObject> cellMap = new Dictionary<string, GameObject>();

    //[Header("细胞类型模式")]
    //public Color[] typeColors;

    [Header("图例面板")]
    public GameObject legendPanel;  // 图例面板引用
    public GameObject legendItemPrefab;  // 图例条目预制体引用
    public Transform legendContent;  // 包含图例条目的容器

    [Header("语义分割分区")]
    public TMP_Dropdown regionDropdown; // 拖入刚才生成的 Dropdown

    private List<string> currentRegionNames = new List<string>();
    private List<int> savedRegionIds = new List<int>();

    public enum ViewMode
    {
        Expression,
        CellType,
        AI_Annotation,
        TissueRegion
    }
    public ViewMode currentMode = ViewMode.Expression;

    private Dictionary<string, CellData> currentDataMap = new Dictionary<string, CellData>();
    private Dictionary<string, int> aiPredictionMap = new Dictionary<string, int>();

    public int highlightedTypeID = -1; // -1 显示所有
    public List<string> annotationLegend = new List<string>();
    // 添加图例条目对象列表，便于清理
    private List<GameObject> legendItems = new List<GameObject>();

    struct CellData
    {
        public string id;
        public float x;
        public float y;
        public float expression;
        public int typeId;
        public string typeName;
    }
    [Header("色彩系统设置")]
    [Range(1, 100)]
    public int typeColorCount = 45; // 在 Unity Inspector 中设置
    public Color[] typeColors;      // 运行时会自动填充

    [Header("可视化配置")]
    public float saturation = 0.8f; // 饱和度：颜色鲜艳程度
    public float brightness = 0.9f; // 亮度
    void Awake()
    {
        GenerateTypeColors();
    }

    // 自动生成颜色的核心代码
    public void GenerateTypeColors()
    {
        typeColors = new Color[typeColorCount];

        for (int i = 0; i < typeColorCount; i++)
        {
            // 将 H (色相) 在 0 到 1 之间均匀切分
            // 0 为红色，0.33 为绿色，0.66 为蓝色，1 又回到红色
            float hue = (float)i / typeColorCount;

            // 使用 Unity 内置函数从 HSV 转回 RGB
            typeColors[i] = Color.HSVToRGB(hue, saturation, brightness);
        }

        Debug.Log($"[Unity] 已自动生成 {typeColorCount} 种区分颜色。");
    }
    void Start()
    {
        string filePath = Path.Combine(Application.streamingAssetsPath, csvFileName);
        if (File.Exists(filePath))
        {
            List<CellData> dataList = ParseCSV(filePath);
            SpawnCells(dataList);
        }
        else
        {
            Debug.LogError("找不到CSV文件！" + filePath);
        }

        // 初始化时隐藏图例面板
        if (legendPanel != null)
            legendPanel.SetActive(false);
    }

    List<CellData> ParseCSV(string path)
    {
        List<CellData> list = new List<CellData>();
        string[] lines = File.ReadAllLines(path);

        for (int i = 1; i < lines.Length; i++)
        {
            string line = lines[i];
            if (string.IsNullOrEmpty(line)) continue;
            string[] values = line.Split(',');
            if (values.Length < 6) continue;

            CellData data = new CellData();
            try
            {
                data.id = values[0];
                data.x = float.Parse(values[1], CultureInfo.InvariantCulture);
                data.y = float.Parse(values[2], CultureInfo.InvariantCulture);
                data.expression = float.Parse(values[4], CultureInfo.InvariantCulture);
                data.typeId = int.Parse(values[5]);
                if (values.Length > 6) data.typeName = values[6];

                list.Add(data);

                if (!currentDataMap.ContainsKey(data.id))
                {
                    currentDataMap.Add(data.id, data);
                }
            }
            catch (System.Exception e) { Debug.LogWarning(e.Message); }
        }
        return list;
    }
    void SpawnCells(List<CellData> cells)
    {
        GameObject root = new GameObject("Cell_Container");
        root.transform.position = Vector3.zero;
        MaterialPropertyBlock props = new MaterialPropertyBlock();

        foreach (var cell in cells)
        {
            GameObject obj = Instantiate(cellPrefab, root.transform);
            obj.name = cell.id;


            if (!cellMap.ContainsKey(cell.id)) cellMap.Add(cell.id, obj);

            UpdateObjectVisuals(obj, cell, props);
        }
    }


    void UpdateObjectVisuals(GameObject obj, CellData cell, MaterialPropertyBlock props, bool isImputation = false, float previousExpr = 0f)
    {
        float targetValue = 0f;
        Color baseColor = Color.white;
        float scale = 0.5f;

        // --- 模式分支 ---
        if (currentMode == ViewMode.Expression)
        {
            targetValue = cell.expression;
            baseColor = colorGradient.Evaluate(cell.expression);
            scale = 0.5f + cell.expression;
        }
        else if (currentMode == ViewMode.CellType)
        {
            targetValue = 1.0f;
            int safeId = Mathf.Clamp(cell.typeId, 0, typeColors.Length - 1);
            baseColor = typeColors[safeId];
            scale = 0.5f;
        }
        else if (currentMode == ViewMode.AI_Annotation)
        {
            targetValue = 0.5f;
            int predId = 0;

            // 获取预测 ID，如果没预测数据，视为 ID 0 或其他处理
            if (aiPredictionMap.ContainsKey(cell.id))
            {
                predId = aiPredictionMap[cell.id];
            }

            // 判断是否需要显示
            // 如果 highlightedTypeID == -1，显示所有
            // 如果 predId == highlightedTypeID，显示该类型
            if (highlightedTypeID == -1 || predId == highlightedTypeID)
            {
                // 显示状态
                int safeId = Mathf.Clamp(predId, 0, typeColors.Length - 1);
                baseColor = typeColors[safeId];
                scale = 0.8f;
            }
            else
            {
                // 直接把 Scale 设为 0，让它彻底消失
                scale = 0.0f;
            }
        }

        Vector3 targetPos = new Vector3(
            cell.x * positionScale,
            targetValue * heightMultiplier,
            cell.y * positionScale
        );

        if (isImputation && cell.expression > previousExpr + 0.05f)
        {
            StartCoroutine(AnimateGrowth(obj, targetPos, scale, cell.expression, props));
        }
        else
        {
            obj.transform.position = targetPos;
            obj.transform.localScale = Vector3.one *  CellScale * scale; // 如果 scale 是 0，它就看不见了

            props.SetColor("_BaseColor", baseColor);
            props.SetColor("_EmissionColor", baseColor * emissionIntensity);

            obj.GetComponent<Renderer>().SetPropertyBlock(props);
        }
    }

    IEnumerator AnimateGrowth(GameObject obj, Vector3 targetPos, float targetScale, float expressionValue, MaterialPropertyBlock props)
    {
        float duration = 1.5f;
        float timer = 0f;

        Vector3 startPos = obj.transform.position;
        Vector3 startScale = obj.transform.localScale;

        Renderer rend = obj.GetComponent<Renderer>();

        while (timer < duration)
        {
            timer += Time.deltaTime;
            float t = timer / duration;
            t = Mathf.Sin(t * Mathf.PI * 0.5f);

            obj.transform.position = Vector3.Lerp(startPos, targetPos, t);
            obj.transform.localScale = Vector3.Lerp(startScale, Vector3.one * targetScale, t);

            if (rend != null)
            {
                float flash = Mathf.PingPong(Time.time * 5.0f, 1.0f);
                Color magicColor = Color.cyan;
                Color finalColor = Color.Lerp(magicColor, colorGradient.Evaluate(expressionValue), t);
                props.SetColor("_BaseColor", finalColor);
                props.SetColor("_EmissionColor", finalColor * (3.0f + flash * 5.0f));
                rend.SetPropertyBlock(props);
            }
            yield return null;
        }
        Color c = colorGradient.Evaluate(expressionValue);
        props.SetColor("_BaseColor", c);
        props.SetColor("_EmissionColor", c * emissionIntensity);
        rend.SetPropertyBlock(props);
    }

    [System.Serializable]
    public class UpdateData { public string id; public float new_expr; }

    [System.Serializable]
    public class ServerResponse
    {
        public string status;
        public string message;
        public UpdateData[] updates;
    }

    public void UpdateVisuals(string jsonResponse)
    {
        ServerResponse response = JsonUtility.FromJson<ServerResponse>(jsonResponse);
        if (response == null || response.updates == null) return;

        MaterialPropertyBlock props = new MaterialPropertyBlock();

        bool isImputationAnim = false;
        if (!string.IsNullOrEmpty(response.message))
        {
            if (response.message.Contains("Imputation"))
            {
                Debug.Log("包含基因插补Imputation关键字");
            }
            else
            {
                Debug.Log($"不包含基因插补Imputation关键字，信息为{response.message}");
            }
            isImputationAnim = response.message.Contains("Imputation") || response.message.Contains("Denoise");
        }

        foreach (var update in response.updates)
        {
            if (cellMap.ContainsKey(update.id))
            {
                float oldExpr = 0f;
                if (currentDataMap.ContainsKey(update.id)) oldExpr = currentDataMap[update.id].expression;

                if (currentDataMap.ContainsKey(update.id))
                {
                    CellData data = currentDataMap[update.id];
                    data.expression = update.new_expr;
                    currentDataMap[update.id] = data;
                }

                GameObject obj = cellMap[update.id];
                UpdateObjectVisuals(obj, currentDataMap[update.id], props, isImputationAnim, oldExpr);
            }
        }
        Debug.Log($"成功更新了 {response.updates.Length} 个细胞的视觉状态！");
    }

    [System.Serializable]
    public class AnnotationUpdate { public string id; public int pred_id; }
    [System.Serializable]
    public class AnnotationResponse { public string status; public string[] legend; public AnnotationUpdate[] updates; }

    public void ApplyAnnotationData(string jsonResponse)
    {
        AnnotationResponse res = JsonUtility.FromJson<AnnotationResponse>(jsonResponse);
        if (res.status != "success") return;

        annotationLegend.Clear();
        annotationLegend.AddRange(res.legend);

        foreach (var update in res.updates)
        {
            if (aiPredictionMap.ContainsKey(update.id))
                aiPredictionMap[update.id] = update.pred_id;
            else
                aiPredictionMap.Add(update.id, update.pred_id);
        }

        currentMode = ViewMode.AI_Annotation;
        highlightedTypeID = -1;
        RefreshAllCells();

        // 获取完整的图例信息
        StartCoroutine(FetchAnnotationLegend((success) =>
        {
            if (!success)
            {
                Debug.LogError("Failed to fetch annotation legend");
            }
        }));
    }

    public void SwitchMode(int modeIndex)
    {
        ViewMode oldMode = currentMode;
        currentMode = (ViewMode)modeIndex;

        // 如果从AI注释模式切换出去，隐藏图例面板
        if (oldMode == ViewMode.AI_Annotation && currentMode != ViewMode.AI_Annotation)
        {
            ClearLegendPanel();
        }
        // 如果切换到AI注释模式，显示图例面板
        else if (currentMode == ViewMode.AI_Annotation)
        {
            // 只有在已经有注释数据的情况下才尝试显示图例
            if (annotationLegend.Count > 0)
            {
                StartCoroutine(FetchAnnotationLegend((success) =>
                {
                    if (!success)
                    {
                        Debug.LogError("Failed to fetch annotation legend");
                    }
                }));
            }
        }

        RefreshAllCells();
    }

    public void ToggleViewMode() { int nextMode = (currentMode == ViewMode.Expression) ? 1 : 0; SwitchMode(nextMode); }

    void RefreshAllCells()
    {
        MaterialPropertyBlock props = new MaterialPropertyBlock();
        foreach (var kvp in currentDataMap)
        {
            if (cellMap.ContainsKey(kvp.Key)) UpdateObjectVisuals(cellMap[kvp.Key], kvp.Value, props);
        }
    }

    public bool GetCellDetails(string id, out string typeName, out Vector2 pos, out float expr)
    {
        if (currentDataMap.ContainsKey(id))
        {
            CellData data = currentDataMap[id];
            typeName = string.IsNullOrEmpty(data.typeName) ? "Unknown" : data.typeName;
            pos = new Vector2(data.x, data.y);
            expr = data.expression;
            return true;
        }
        typeName = "Unknown"; pos = Vector2.zero; expr = 0;
        return false;
    }

    public float GetAverageExpression()
    {
        if (currentDataMap.Count == 0) return 0;
        float sum = 0;
        foreach (var kvp in currentDataMap) sum += kvp.Value.expression;
        return sum / currentDataMap.Count;
    }

    // 添加辅助类用于解析响应
    [System.Serializable]
    public class LegendItem
    {
        public int id;
        public string name;
    }

    [System.Serializable]
    public class LegendResponse
    {
        public string status;
        public LegendItem[] legend;
    }

    /// <summary>
    /// IEnumerator：这是 C# 中用于实现"协程（Coroutine）"返回类型的接口。在 Unity 中，
    /// 协程允许你以非阻塞的方式执行耗时操作（比如网络请求），并在特定时间点暂停或恢复执行。
    /// </summary>
    /// <param name="onComplete"></param>
    /// <returns></returns>
    public IEnumerator FetchAnnotationLegend(System.Action<bool> onComplete)  // 获取注解图例数据
    {
        UnityWebRequest request = UnityWebRequest.Get("http://localhost:8000/annotation_legend");
        yield return request.SendWebRequest(); //协程的关键字。它会暂停当前方法的执行，直到 SendWebRequest() 返回的结果（通常是 AsyncOperation）完成。

        if (request.result == UnityWebRequest.Result.Success)
        {
            //request.downloadHandler：处理从服务器下载的数据。默认是 DownloadHandlerBuffer，会把响应体存为字符串。
            //.text：获取响应的文本内容（通常是 JSON 格式）。
            string jsonRespone = request.downloadHandler.text;
            ProcessLegendData(jsonRespone);

            /*检查回调 onComplete 是否被提供（避免空引用异常）。
            如果提供了，就调用它，并传入 true 表示操作成功。
            这样，调用者可以执行后续逻辑，比如隐藏加载动画、显示成功提示等。*/
            if (onComplete != null) onComplete(true);
        }
        else
        {
            Debug.LogError("获取图例数据失败：" + request.error);
            if (onComplete != null) onComplete(false);
        }
    }

    private void ProcessLegendData(string jsonRespone)
    {
        //JsonUtility.FromJson<T>(...)：Unity 内置的 JSON 反序列化工具，将 JSON 字符串转换为指定类型 T 的 C# 对象
        var response = JsonUtility.FromJson<LegendResponse>(jsonRespone);
        if (response.status == "success")
        {
            // 更新图例数据
            CreateLegendPanel(response.legend);
        }
    }

    private void CreateLegendPanel(LegendItem[] legendData)
    {
        // 清理旧的
        ClearLegendPanel();

        // 显示图例面板
        if (legendPanel != null)
            legendPanel.SetActive(true);

        foreach (var item in legendData)
        {
            if (legendItemPrefab != null && legendContent != null)
            {
                GameObject legendItemObj = Instantiate(legendItemPrefab, legendContent);
                legendItems.Add(legendItemObj);

                // 获取子组件 - 支持TextMeshPro
                Image colorBox = null;
                Component labelComponent = null; // 使用通用组件类型
                TMPro.TMP_Text tmpTextLabel = null; // TMP文本组件
                Text uiTextLabel = null; // 标准UI文本组件

                // 尝试通过名称查找ColorBox
                Transform colorBoxTransform = legendItemObj.transform.Find("ColorBox");
                if (colorBoxTransform != null)
                {
                    colorBox = colorBoxTransform.GetComponent<Image>();
                }
                else
                {
                    // 如果按名称找不到，尝试获取第一个Image组件
                    Image[] images = legendItemObj.GetComponentsInChildren<Image>();
                    foreach (var img in images)
                    {
                        if (img.gameObject != legendItemObj) // 不是自己的Image组件
                        {
                            colorBox = img;
                            break;
                        }
                    }
                }

                // 尝试通过名称查找Label
                Transform labelTransform = legendItemObj.transform.Find("Label");
                if (labelTransform != null)
                {
                    // 检查是否是TMP文本组件
                    tmpTextLabel = labelTransform.GetComponent<TMPro.TMP_Text>();
                    if (tmpTextLabel == null)
                    {
                        // 如果不是TMP组件，检查是否是标准UI文本组件
                        uiTextLabel = labelTransform.GetComponent<Text>();
                    }
                }
                else
                {
                    // 如果按名称找不到，尝试获取文本组件
                    tmpTextLabel = legendItemObj.GetComponentInChildren<TMPro.TMP_Text>();
                    if (tmpTextLabel == null || tmpTextLabel.gameObject == legendItemObj)
                    {
                        tmpTextLabel = null;
                        // 尝试查找标准UI文本组件
                        uiTextLabel = legendItemObj.GetComponentInChildren<Text>();
                        if (uiTextLabel == null || uiTextLabel.gameObject == legendItemObj)
                        {
                            uiTextLabel = null;
                        }
                    }
                }

                // 设置颜色
                if (colorBox != null && item.id < typeColors.Length)
                {
                    // 确保颜色alpha值不为0
                    Color finalColor = typeColors[item.id];
                    if (finalColor.a <= 0f)
                    {
                        finalColor.a = 1f; // 设置为不透明
                        Debug.Log("修正颜色Alpha值为1.0");
                    }

                    colorBox.color = finalColor;
                    Debug.Log("设置颜色: " + finalColor + " 对于类型ID: " + item.id);
                }
                else
                {
                    Debug.LogWarning("未能找到ColorBox组件或类型ID超出范围: " + item.id);
                }

                // 设置文本（支持TMP和标准Text）
                if (tmpTextLabel != null)
                {
                    tmpTextLabel.text = item.name;
                    Debug.Log("设置TMP标签: " + item.name + " 对于类型ID: " + item.id);
                }
                else if (uiTextLabel != null)
                {
                    uiTextLabel.text = item.name;
                    Debug.Log("设置UI标签: " + item.name + " 对于类型ID: " + item.id);
                }
                else
                {
                    Debug.LogWarning("未能找到Label文本组件 对于类型ID: " + item.id);
                }
            }
        }
        Canvas.ForceUpdateCanvases();
        if (legendContent.TryGetComponent<VerticalLayoutGroup>(out var layout))
        {
            layout.enabled = false;
            layout.enabled = true; // 切换一下开关可强制刷新布局
        }
    }
    private void ClearLegendPanel()
    {
        foreach (var item in legendItems)
        {
            if (item != null)
                DestroyImmediate(item);
        }
        legendItems.Clear();

        // 隐藏图例面板
        if (legendPanel != null)
            legendPanel.SetActive(false);
    }


    public void ApplyRegionSegmentation(List<int> regionIds, List<string> regionNames)
    {
        currentMode = ViewMode.TissueRegion;
        Debug.Log($"[Unity] 语义分割染色与平面化对齐开始，数据量: {regionIds.Count}");

        MaterialPropertyBlock propBlock = new MaterialPropertyBlock();
        int colorID = Shader.PropertyToID("_BaseColor");

        // 设定一个统一的高度中间值
        float flatY = 0f;

        int index = 0;
        foreach (var kvp in cellMap)
        {
            if (index >= regionIds.Count) break;

            GameObject cellObj = kvp.Value;

            // --- 核心修改：强制对齐到平面 ---
            // 保持原始的 X 和 Z（水平位置），将 Y（高度）统一
            Vector3 currentPos = cellObj.transform.localPosition;
            cellObj.transform.localPosition = new Vector3(currentPos.x, flatY, currentPos.z);
            // ------------------------------

            MeshRenderer mr = cellObj.GetComponent<MeshRenderer>();
            if (mr != null)
            {
                int rId = regionIds[index];
                Color targetColor = typeColors[rId % typeColors.Length];

                mr.GetPropertyBlock(propBlock);
                propBlock.SetColor(colorID, targetColor);
                mr.SetPropertyBlock(propBlock);

                // 放大一点，让区域色块更连续（减少缝隙）
                cellObj.transform.localScale = Vector3.one * 1.5f;
            }
            index++;
        }
        Debug.Log("[Unity] 平面化对齐与区域染色完成。");

        if (regionNames != null && regionNames.Count > 0)
        {
            // 构造图例数据结构
            LegendItem[] legendData = new LegendItem[regionNames.Count];
            for (int i = 0; i < regionNames.Count; i++)
            {
                legendData[i] = new LegendItem
                {
                    id = i,
                    name = regionNames[i]
                };
            }
            // 直接调用你现有的生成函数
            CreateLegendPanel(legendData);
        }
        currentRegionNames = regionNames;
        savedRegionIds = regionIds;
        InitRegionDropdown(regionNames);
    }
    private void InitRegionDropdown(List<string> names)
    {
        if (regionDropdown == null) return;

        regionDropdown.ClearOptions();

        // 添加一个“显示全部”的选项
        List<string> options = new List<string> { "Show All" };
        options.AddRange(names);

        regionDropdown.AddOptions(options);

        // 绑定点击事件
        regionDropdown.onValueChanged.RemoveAllListeners();
        regionDropdown.onValueChanged.AddListener(OnDropdownValueChanged);
    }

    // 当 Dropdown 切换时触发
    private void OnDropdownValueChanged(int index)
    {
        // index 为 0 代表 "Show All"
        // index > 0 代表具体的分区，其 ID 对应 index - 1
        FilterRegions(index - 1);
    }
    // regionIdToDisplay 为 -1 时显示全部，否则显示特定 ID
    public void FilterRegions(int targetRegionId)
    {
        Debug.Log($"[Unity] 过滤分区，目标 ID: {targetRegionId}");

        // 我们需要知道每个细胞属于哪个分区
        // 假设你在之前的 ApplyRegionSegmentation 中已经把分区 ID 存到了某处
        // 如果没有存，我们需要在 ApplyRegionSegmentation 时给 GameObject 一个组件或标识

        int index = 0;
        foreach (var kvp in cellMap)
        {
            // 注意：这里需要 regionIds 数组，建议将其设为类成员变量
            int cellRegionId = savedRegionIds[index];
            GameObject cellObj = kvp.Value;

            if (targetRegionId == -1 || cellRegionId == targetRegionId)
            {
                cellObj.SetActive(true); // 显示
                                         // 或者使用之前的 scale = 1.0f 逻辑，如果你不想完全隐藏
            }
            else
            {
                cellObj.SetActive(false); // 隐藏
            }
            index++;
        }
    }
}