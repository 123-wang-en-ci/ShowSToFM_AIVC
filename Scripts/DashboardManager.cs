using UnityEngine;
using UnityEngine.UI; 
using TMPro;
using System.Linq; 

public class DashboardManager : MonoBehaviour
{
    [Header("UI设置")]
    public RectTransform barCurrent; 
    public RectTransform barAverage; 
    public TextMeshProUGUI valCurrentText;
    public TextMeshProUGUI valAverageText;

    public float maxHeight = 200f; 

    public static DashboardManager Instance;

    void Awake() { Instance = this; }


    public void UpdateChart(float currentVal, float allCellsAverage)
    {
        valCurrentText.text = currentVal.ToString("F2");
        valAverageText.text = allCellsAverage.ToString("F2");

        barCurrent.sizeDelta = new Vector2(barCurrent.sizeDelta.x, currentVal * maxHeight);
        barAverage.sizeDelta = new Vector2(barAverage.sizeDelta.x, allCellsAverage * maxHeight);
    }
}