/*using UnityEngine;
using UnityEngine.Networking; // 用于联网
using System.Collections;
using System.Text;
using UnityEngine.UI;

// 定义发送给 Python 的数据格式
[System.Serializable]
public class PerturbRequest
{
    public string target_id;
    public string perturb_type; // "KO" or "OE"
    public string target_gene;  // "TP53"
}

// 基因查询请求结构体
[System.Serializable]
public class GeneRequest
{
    public string gene_name;
    public bool use_imputation; // 告诉后端是否开启插补
}

public class InteractionManager : MonoBehaviour
{
    [Header("设置")]
    public string serverUrl = "http://127.0.0.1:8000/perturb"; // Python 服务器地址
    public DataLoader dataLoader; // 引用 DataLoader 脚本
    public Camera mainCamera;     // 摄像机

    public enum InteractionMode { Inspect, Perturb }
    public InteractionMode currentMode = InteractionMode.Inspect;

    [Header("模式按钮")]
    public Image btnInspectImg;
    public Image btnPerturbImg;
    public Color activeColor = Color.green;
    public Color inactiveColor = Color.white;

    [Header("扰动参数 UI")]
    public TMPro.TMP_InputField perturbGeneInput;
    public Toggle toggleKO;

    // [修改] 移除了全量插补 Toggle，保留此变量记录当前基因
    private string lastSearchedGene = "RESET";

    [Header("注释 UI")]
    public TMPro.TMP_Dropdown typeDropdown; // 拖入 Dropdown_CellTypes

    public void SetInspectMode()
    {
        currentMode = InteractionMode.Inspect;
        UpdateButtonVisuals();
        Debug.Log("切换模式: 仅检视信息");
    }

    public void SetPerturbMode()
    {
        currentMode = InteractionMode.Perturb;
        UpdateButtonVisuals();
        Debug.Log("切换模式: 开启扰动推演");
    }

    // =========================================================
    // 【修改】智能插补按钮逻辑 (仅单基因)
    // =========================================================
    public void RequestImputation()
    {
        // 只能针对具体基因进行插补，不能针对 RESET 视图
        if (string.IsNullOrEmpty(lastSearchedGene) || lastSearchedGene == "RESET")
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Please search a specific gene first.", true);
            return;
        }

        Debug.Log($"[UI] Requesting Single Gene Imputation for: {lastSearchedGene}");
        // 发送基因查询请求，带上 use_imputation = true
        StartCoroutine(SendGeneSwitchRequest(lastSearchedGene, true));
    }

    // =========================================================
    // 【新增】保存当前插补数据按钮
    // =========================================================
    public void RequestSaveImputation()
    {
        if (string.IsNullOrEmpty(lastSearchedGene) || lastSearchedGene == "RESET")
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("No gene data to save.", true);
            return;
        }
        StartCoroutine(SendSaveImputationRequest());
    }

    IEnumerator SendSaveImputationRequest()
    {
        // 调用 /save_imputation 接口
        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/save_imputation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();

        // 告诉后端要保存哪个基因
        GeneRequest req = new GeneRequest { gene_name = lastSearchedGene, use_imputation = true };
        byte[] bodyRaw = Encoding.UTF8.GetBytes(JsonUtility.ToJson(req));

        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            // 解析返回的消息
            string jsonString = request.downloadHandler.text;
            // 这里假设后端返回标准的 {status, message} 格式
            DataLoader.ServerResponse response = JsonUtility.FromJson<DataLoader.ServerResponse>(jsonString);

            string msg = string.IsNullOrEmpty(response.message) ? "Imputation Saved!" : response.message;
            bool isError = response.status != "success";

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage(msg, isError);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Save Failed: " + request.error, true);
        }
    }

    // 发送基因查询/单基因插补请求 (/switch_gene)
    IEnumerator SendGeneSwitchRequest(string geneName, bool doImpute)
    {
        GeneRequest req = new GeneRequest
        {
            gene_name = geneName,
            use_imputation = doImpute
        };

        string json = JsonUtility.ToJson(req);

        UnityWebRequest request = new UnityWebRequest("http://127.0.0.1:8000/switch_gene", "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoader.ServerResponse response = JsonUtility.FromJson<DataLoader.ServerResponse>(jsonString);

            if (!string.IsNullOrEmpty(response.message))
            {
                bool isError = (response.status != "success");
                if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage(response.message, isError);
            }

            if (response.updates != null && response.updates.Length > 0)
            {
                lastSearchedGene = geneName;
                dataLoader.UpdateVisuals(jsonString);
                if (dataLoader.currentMode != DataLoader.ViewMode.Expression)
                    dataLoader.SwitchMode((int)DataLoader.ViewMode.Expression);
            }
        }
        else
        {
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage("Request Failed", true);
        }
    }

    // 普通基因查询入口 (UI_GeneSearch 调用此方法)
    public void RequestGeneSwitch(string geneName)
    {
        // 普通搜索不带插补
        StartCoroutine(SendGeneSwitchRequest(geneName, false));
    }

    // 关闭插补 (回到原始数据) - 其实就是重新查一次不带插补的
    public void RequestDisableImputation()
    {
        if (!string.IsNullOrEmpty(lastSearchedGene))
        {
            StartCoroutine(SendGeneSwitchRequest(lastSearchedGene, false));
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage("Showing Raw Data.", false);
        }
    }

    // ... (Perturb 逻辑保持不变) ...
    IEnumerator SendPerturbRequest(string id)
    {
        string pType = toggleKO.isOn ? "KO" : "OE";
        string pGene = "";
        if (perturbGeneInput != null && !string.IsNullOrEmpty(perturbGeneInput.text) && !string.IsNullOrWhiteSpace(perturbGeneInput.text))
        {
            pGene = perturbGeneInput.text.Trim();
        }
        else
        {
            string errorMsg = "Input Error: Please enter a Gene Symbol (e.g. NPHS1).";
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage(errorMsg, true);
            yield break;
        }

        PerturbRequest req = new PerturbRequest
        {
            target_id = id,
            perturb_type = pType,
            target_gene = pGene
        };

        string json = JsonUtility.ToJson(req);

        UnityWebRequest request = new UnityWebRequest(serverUrl, "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoader.ServerResponse response = JsonUtility.FromJson<DataLoader.ServerResponse>(jsonString);

            if (!string.IsNullOrEmpty(response.message))
            {
                bool isError = (response.status != "success");
                if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage(response.message, isError);
            }

            if (response.updates != null && response.updates.Length > 0)
            {
                dataLoader.UpdateVisuals(jsonString);
            }
        }
        else
        {
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage("Server Connection Failed", true);
        }
    }

    // ... (其他 Save/Clear 逻辑保持不变) ...
    public void RequestManualSave()
    {
        StartCoroutine(SendSaveRequest());
    }

    IEnumerator SendSaveRequest()
    {
        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/save_manual"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();
        byte[] bodyRaw = Encoding.UTF8.GetBytes("{}");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoader.ServerResponse response = JsonUtility.FromJson<DataLoader.ServerResponse>(jsonString);
            string msg = string.IsNullOrEmpty(response.message) ? "Snapshot Saved" : response.message;

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage(msg, false);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Save Failed: " + request.error, true);
        }
    }

    public void RequestClearData()
    {
        StartCoroutine(SendClearRequest());
    }

    IEnumerator SendClearRequest()
    {
        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/clear_perturbation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();
        byte[] bodyRaw = Encoding.UTF8.GetBytes("{}");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            dataLoader.UpdateVisuals(request.downloadHandler.text);
            SetInspectMode();
            lastSearchedGene = "RESET";

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Reset Successful", false);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Reset Failed", true);
        }
    }

    void UpdateButtonVisuals()
    {
        if (btnInspectImg != null && btnPerturbImg != null)
        {
            btnInspectImg.color = (currentMode == InteractionMode.Inspect) ? activeColor : inactiveColor;
            btnPerturbImg.color = (currentMode == InteractionMode.Perturb) ? activeColor : inactiveColor;
        }
    }

    void Start()
    {
        if (mainCamera == null) mainCamera = Camera.main;
        UpdateButtonVisuals();
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            HandleClick();
        }
    }

    void HandleClick()
    {
        Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit))
        {
            string clickedId = hit.transform.name;

            // 更新 UI 面板
            string typeName;
            Vector2 pos;
            float currentExpr;
            if (dataLoader.GetCellDetails(clickedId, out typeName, out pos, out currentExpr))
            {
                float avgExpr = dataLoader.GetAverageExpression();
                if (UIManager.Instance != null)
                    UIManager.Instance.ShowCellDetails(clickedId, typeName, pos, currentExpr, avgExpr);
                if (DashboardManager.Instance != null)
                    DashboardManager.Instance.UpdateChart(currentExpr, avgExpr);
            }

            if (currentMode == InteractionMode.Perturb)
            {
                StartCoroutine(SendPerturbRequest(clickedId));
            }
        }
    }

    // 绑定给 "AI Annotation" 按钮
    public void RequestAnnotation()
    {
        StartCoroutine(SendAnnotationRequest());
    }

    IEnumerator SendAnnotationRequest()
    {
        if (UIManager.Instance != null)
            UIManager.Instance.ShowSystemMessage("AI Predicting Cell Types...", false);

        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/get_annotation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();
        request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes("{}"));
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            // 1. 应用数据
            dataLoader.ApplyAnnotationData(request.downloadHandler.text);

            // 2. 初始化下拉框
            if (typeDropdown != null)
            {
                typeDropdown.gameObject.SetActive(true); // 显示下拉框
                typeDropdown.ClearOptions();

                // 添加 "All Types" 选项
                System.Collections.Generic.List<string> options = new System.Collections.Generic.List<string>();
                options.Add("Show All Types");
                options.AddRange(dataLoader.annotationLegend); // 添加后端传来的具体类型

                typeDropdown.AddOptions(options);
                typeDropdown.value = 0; // 默认选 All

                // 绑定下拉框事件 (注意防止重复绑定)
                typeDropdown.onValueChanged.RemoveAllListeners();
                typeDropdown.onValueChanged.AddListener(OnTypeDropdownChanged);
            }

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Annotation Complete!", false);
        }
        else
        {
            Debug.LogError(request.error);
        }
    }

    // 下拉框回调
    public void OnTypeDropdownChanged(int index)
    {
        // index 0 是 "Show All" -> ID -1
        // index 1 是 第一个类型 -> ID 0
        int typeId = index - 1;

        Debug.Log($"切换高亮类型: {typeId}");
        dataLoader.highlightedTypeID = typeId;
        dataLoader.SwitchMode((int)DataLoader.ViewMode.AI_Annotation); // 刷新视图
    }

    // 保存细胞注释结果
    public void RequestSaveAnnotation()
    {
        StartCoroutine(SendSaveAnnotationRequest());
    }
    IEnumerator SendSaveAnnotationRequest()
    {
        // 替换 URL 为 /save_annotation
        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/save_annotation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();

        // 发送空 JSON 触发
        byte[] bodyRaw = Encoding.UTF8.GetBytes("{}");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoader.ServerResponse response = JsonUtility.FromJson<DataLoader.ServerResponse>(jsonString);

            string msg = string.IsNullOrEmpty(response.message) ? "Annotation Saved!" : response.message;
            bool isError = response.status != "success";

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage(msg, isError);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Save Failed: " + request.error, true);
        }
    }
}*/
using UnityEngine;
using UnityEngine.Networking; // 用于联网
using System.Collections;
using System.Text;
using UnityEngine.UI;
using System.Collections.Generic;

// 定义发送给 Python 的数据格式
[System.Serializable]
public class PerturbRequest
{
    public string target_id;
    public string perturb_type; // "KO" or "OE"
    public string target_gene;  // "TP53"
}

// 基因查询请求结构体
[System.Serializable]
public class GeneRequest
{
    public string gene_name;
    public bool use_imputation; // 告诉后端是否开启插补
}

// 组织区域语义分割
[System.Serializable]
public class RegionResponse
{
    public string status;
    public List<int> regions;
    public List<string> names;
}
public class InteractionManager : MonoBehaviour
{
    [Header("设置")]
    public string serverUrl = "http://127.0.0.1:8000/perturb"; // Python 服务器地址
    public DataLoader dataLoader; // 引用 DataLoader 脚本
    public Camera mainCamera;     // 摄像机

    public enum InteractionMode { Inspect, Perturb }
    public InteractionMode currentMode = InteractionMode.Inspect;

    [Header("模式按钮")]
    public Image btnInspectImg;
    public Image btnPerturbImg;
    public Color activeColor = Color.green;
    public Color inactiveColor = Color.white;

    [Header("扰动参数 UI")]
    public TMPro.TMP_InputField perturbGeneInput;
    public Toggle toggleKO;

    // [修改] 移除了全量插补 Toggle，保留此变量记录当前基因
    private string lastSearchedGene = "RESET";

    [Header("注释 UI")]
    public TMPro.TMP_Dropdown typeDropdown; // 拖入 Dropdown_CellTypes

    public void SetInspectMode()
    {
        currentMode = InteractionMode.Inspect;
        UpdateButtonVisuals();
        Debug.Log("切换模式: 仅检视信息");
    }

    public void SetPerturbMode()
    {
        currentMode = InteractionMode.Perturb;
        UpdateButtonVisuals();
        Debug.Log("切换模式: 开启扰动推演");
    }

    // =========================================================
    // 智能插补按钮逻辑 (仅单基因)
    // =========================================================
    public void RequestImputation()
    {
        // 只能针对具体基因进行插补，不能针对 RESET 视图
        if (string.IsNullOrEmpty(lastSearchedGene) || lastSearchedGene == "RESET")
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Please search a specific gene first.", true);
            return;
        }

        Debug.Log($"[UI] Requesting Single Gene Imputation for: {lastSearchedGene}");
        // 发送基因查询请求，带上 use_imputation = true
        StartCoroutine(SendGeneSwitchRequest(lastSearchedGene, true));
    }

    // =========================================================
    // 保存当前插补数据按钮
    // =========================================================
    public void RequestSaveImputation()
    {
        if (string.IsNullOrEmpty(lastSearchedGene) || lastSearchedGene == "RESET")
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("No gene data to save.", true);
            return;
        }
        StartCoroutine(SendSaveImputationRequest());
    }

    IEnumerator SendSaveImputationRequest()
    {
        // 调用 /save_imputation 接口
        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/save_imputation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();

        // 告诉后端要保存哪个基因
        GeneRequest req = new GeneRequest { gene_name = lastSearchedGene, use_imputation = true };
        byte[] bodyRaw = Encoding.UTF8.GetBytes(JsonUtility.ToJson(req));

        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            // 解析返回的消息
            string jsonString = request.downloadHandler.text;
            // 这里假设后端返回标准的 {status, message} 格式
            DataLoader.ServerResponse response = JsonUtility.FromJson<DataLoader.ServerResponse>(jsonString);

            string msg = string.IsNullOrEmpty(response.message) ? "Imputation Saved!" : response.message;
            bool isError = response.status != "success";

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage(msg, isError);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Save Failed: " + request.error, true);
        }
    }

    // 发送基因查询/单基因插补请求 (/switch_gene)
    IEnumerator SendGeneSwitchRequest(string geneName, bool doImpute)
    {
        GeneRequest req = new GeneRequest
        {
            gene_name = geneName,
            use_imputation = doImpute
        };

        string json = JsonUtility.ToJson(req);

        UnityWebRequest request = new UnityWebRequest("http://127.0.0.1:8000/switch_gene", "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoader.ServerResponse response = JsonUtility.FromJson<DataLoader.ServerResponse>(jsonString);

            if (!string.IsNullOrEmpty(response.message))
            {
                bool isError = (response.status != "success");
                if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage(response.message, isError);
            }

            if (response.updates != null && response.updates.Length > 0)
            {
                lastSearchedGene = geneName;
                dataLoader.UpdateVisuals(jsonString);
                if (dataLoader.currentMode != DataLoader.ViewMode.Expression)
                    dataLoader.SwitchMode((int)DataLoader.ViewMode.Expression);
            }
        }
        else
        {
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage("Request Failed", true);
        }
    }

    // 普通基因查询入口 (UI_GeneSearch 调用此方法)
    public void RequestGeneSwitch(string geneName)
    {
        // 普通搜索不带插补
        StartCoroutine(SendGeneSwitchRequest(geneName, false));
    }

    // 关闭插补 (回到原始数据) - 其实就是重新查一次不带插补的
    public void RequestDisableImputation()
    {
        if (!string.IsNullOrEmpty(lastSearchedGene))
        {
            StartCoroutine(SendGeneSwitchRequest(lastSearchedGene, false));
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage("Showing Raw Data.", false);
        }
    }

    // ... (Perturb 逻辑保持不变) ...
    IEnumerator SendPerturbRequest(string id)
    {
        string pType = toggleKO.isOn ? "KO" : "OE";
        string pGene = "";
        if (perturbGeneInput != null && !string.IsNullOrEmpty(perturbGeneInput.text) && !string.IsNullOrWhiteSpace(perturbGeneInput.text))
        {
            pGene = perturbGeneInput.text.Trim();
        }
        else
        {
            string errorMsg = "Input Error: Please enter a Gene Symbol (e.g. NPHS1).";
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage(errorMsg, true);
            yield break;
        }

        PerturbRequest req = new PerturbRequest
        {
            target_id = id,
            perturb_type = pType,
            target_gene = pGene
        };

        string json = JsonUtility.ToJson(req);

        UnityWebRequest request = new UnityWebRequest(serverUrl, "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoader.ServerResponse response = JsonUtility.FromJson<DataLoader.ServerResponse>(jsonString);

            if (!string.IsNullOrEmpty(response.message))
            {
                bool isError = (response.status != "success");
                if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage(response.message, isError);
            }

            if (response.updates != null && response.updates.Length > 0)
            {
                dataLoader.UpdateVisuals(jsonString);
            }
        }
        else
        {
            if (UIManager.Instance != null) UIManager.Instance.ShowSystemMessage("Server Connection Failed", true);
        }
    }

    // ... (其他 Save/Clear 逻辑保持不变) ...
    public void RequestManualSave()
    {
        StartCoroutine(SendSaveRequest());
    }

    IEnumerator SendSaveRequest()
    {
        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/save_manual"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();
        byte[] bodyRaw = Encoding.UTF8.GetBytes("{}");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoader.ServerResponse response = JsonUtility.FromJson<DataLoader.ServerResponse>(jsonString);
            string msg = string.IsNullOrEmpty(response.message) ? "Snapshot Saved" : response.message;

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage(msg, false);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Save Failed: " + request.error, true);
        }
    }

    public void RequestClearData()
    {
        StartCoroutine(SendClearRequest());
    }

    IEnumerator SendClearRequest()
    {
        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/clear_perturbation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();
        byte[] bodyRaw = Encoding.UTF8.GetBytes("{}");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            dataLoader.UpdateVisuals(request.downloadHandler.text);
            SetInspectMode();
            lastSearchedGene = "RESET";

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Reset Successful", false);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Reset Failed", true);
        }
    }

    void UpdateButtonVisuals()
    {
        if (btnInspectImg != null && btnPerturbImg != null)
        {
            btnInspectImg.color = (currentMode == InteractionMode.Inspect) ? activeColor : inactiveColor;
            btnPerturbImg.color = (currentMode == InteractionMode.Perturb) ? activeColor : inactiveColor;
        }
    }

    void Start()
    {
        if (mainCamera == null) mainCamera = Camera.main;
        UpdateButtonVisuals();
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            HandleClick();
        }
    }

    void HandleClick()
    {
        Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit))
        {
            string clickedId = hit.transform.name;

            // 更新 UI 面板
            string typeName;
            Vector2 pos;
            float currentExpr;
            if (dataLoader.GetCellDetails(clickedId, out typeName, out pos, out currentExpr))
            {
                float avgExpr = dataLoader.GetAverageExpression();
                if (UIManager.Instance != null)
                    UIManager.Instance.ShowCellDetails(clickedId, typeName, pos, currentExpr, avgExpr);
                if (DashboardManager.Instance != null)
                    DashboardManager.Instance.UpdateChart(currentExpr, avgExpr);
            }

            if (currentMode == InteractionMode.Perturb)
            {
                StartCoroutine(SendPerturbRequest(clickedId));
            }
        }
    }

    // 绑定给 "AI Annotation" 按钮
    public void RequestAnnotation()
    {
        StartCoroutine(SendAnnotationRequest());
    }

    IEnumerator SendAnnotationRequest()
    {
        if (UIManager.Instance != null)
            UIManager.Instance.ShowSystemMessage("AI Predicting Cell Types...", false);

        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/get_annotation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();
        request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes("{}"));
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            // 1. 应用数据
            dataLoader.ApplyAnnotationData(request.downloadHandler.text);

            // 2. 初始化下拉框
            if (typeDropdown != null)
            {
                typeDropdown.gameObject.SetActive(true); // 显示下拉框
                typeDropdown.ClearOptions();

                // 添加 "All Types" 选项
                System.Collections.Generic.List<string> options = new System.Collections.Generic.List<string>();
                options.Add("Show All Types");
                options.AddRange(dataLoader.annotationLegend); // 添加后端传来的具体类型

                typeDropdown.AddOptions(options);
                typeDropdown.value = 0; // 默认选 All

                // 绑定下拉框事件 (注意防止重复绑定)
                typeDropdown.onValueChanged.RemoveAllListeners();
                typeDropdown.onValueChanged.AddListener(OnTypeDropdownChanged);
            }

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Annotation Complete!", false);
        }
        else
        {
            Debug.LogError(request.error);
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Annotation Failed: " + request.error, true);
        }
    }

    // 下拉框回调
    public void OnTypeDropdownChanged(int index)
    {
        // index 0 是 "Show All" -> ID -1
        // index 1 是 第一个类型 -> ID 0
        int typeId = index - 1;

        Debug.Log($"切换高亮类型: {typeId}");
        dataLoader.highlightedTypeID = typeId;
        dataLoader.SwitchMode((int)DataLoader.ViewMode.AI_Annotation); // 刷新视图
    }

    // 保存细胞注释结果
    public void RequestSaveAnnotation()
    {
        StartCoroutine(SendSaveAnnotationRequest());
    }

    IEnumerator SendSaveAnnotationRequest()
    {
        // 替换 URL 为 /save_annotation
        UnityWebRequest request = new UnityWebRequest(serverUrl.Replace("/perturb", "/save_annotation"), "POST");
        request.downloadHandler = new DownloadHandlerBuffer();

        // 发送空 JSON 触发
        byte[] bodyRaw = Encoding.UTF8.GetBytes("{}");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonString = request.downloadHandler.text;
            DataLoader.ServerResponse response = JsonUtility.FromJson<DataLoader.ServerResponse>(jsonString);

            string msg = string.IsNullOrEmpty(response.message) ? "Annotation Saved!" : response.message;
            bool isError = response.status != "success";

            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage(msg, isError);
        }
        else
        {
            if (UIManager.Instance != null)
                UIManager.Instance.ShowSystemMessage("Save Failed: " + request.error, true);
        }
    }

    public void RequestRegionSegmentation()
    {
        StartCoroutine(GetRegionRoutine());
    }
    IEnumerator GetRegionRoutine()
    {
        string url = "http://127.0.0.1:8000/get_tissue_regions";

        // 使用 POST 请求
        UnityWebRequest request = UnityWebRequest.PostWwwForm(url, "");
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string json = request.downloadHandler.text;
            // 使用 JsonUtility 解析时，RegionResponse 类必须打上 [System.Serializable] 标签
            RegionResponse res = JsonUtility.FromJson<RegionResponse>(json);

            if (res.status == "success")
            {
                Debug.Log("收到数据量: " + (res.regions != null ? res.regions.Count.ToString() : "null"));
                dataLoader.ApplyRegionSegmentation(res.regions, res.names);
            }
        }
        else
        {
            Debug.LogError("请求失败: " + request.error);
        }
    }

    // 绑定到“保存分割结果”按钮
    public void OnSaveRegionBtnClick()
    {
        StartCoroutine(SaveRegionDataRoutine());
    }

    IEnumerator SaveRegionDataRoutine()
    {
        Debug.Log("[Unity] 请求后端保存区域分割结果...");
        string url = "http://127.0.0.1:8000/save_tissue_regions";

        using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
        {
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                // 解析返回的 JSON
                var res = JsonUtility.FromJson<CommonResponse>(request.downloadHandler.text);
                Debug.Log($"<color=green>[成功]</color> {res.message}");

                // 可以在 UI 上弹出一个简单的提示框显示“保存成功”
            }
            else
            {
                Debug.LogError($"[失败] 保存请求出错: {request.error}");
            }
        }
    }

    // 辅助类用于解析简单的成功/失败消息
    [System.Serializable]
    public class CommonResponse
    {
        public string status;
        public string message;
    }
}