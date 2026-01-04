using UnityEngine;
using TMPro;

public class UI_GeneSearch : MonoBehaviour
{
    public TMP_InputField inputField;
    public InteractionManager interactionManager;

    // 绑定到 "Search" 按钮
    public void OnSearchClicked()
    {
        string geneName = "";
        if (inputField != null) geneName = inputField.text.Trim();

        if (!string.IsNullOrEmpty(geneName))
        {
            Debug.Log($"[UI] 用户请求搜索基因: {geneName}");
            interactionManager.RequestGeneSwitch(geneName);
        }
        else
        {
            Debug.LogWarning("[UI] 输入框为空！");
        }
    }

    public void OnPreviousViewClicked()
    {
        Debug.Log("[UI] 请求返回默认视图 (View Only, 保留扰动)");
        // 清空输入框，让用户知道现在没搜特定基因
        if (inputField != null) inputField.text = "";

        // 发送 RESET 信号
        interactionManager.RequestGeneSwitch("RESET");
    }
}