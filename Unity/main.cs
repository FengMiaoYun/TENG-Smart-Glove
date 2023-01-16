using System;
using UnityEngine;

public class Finger     //0是初始相对最大角度的比值（增量），1是真实的欧拉角
{
    public float[] MCP_UD = { 0.0f, 0.0f };          //上下移动
    public float[] MCP_LR = { 0.0f, 0.0f };          //左右移动
    public float[] PIP = { 0.0f, 0.0f };
    public float[] DIP = { 0.0f, 0.0f };
}

public class Thumb      //0是初始相对最大角度的比值（增量），1是真实的欧拉角
{
    public float[] trapizum_UD = { 0.0f, 0.0f };     //上下移动
    public float[] trapizum_LR = { 0.0f, 0.0f };     //左右移动
    public float[] MCP_UD = { 0.0f, 0.0f };          //上下移动
    public float[] MCP_LR = { 0.0f, 0.0f };          //左右移动
    public float[] IP = { 0.0f, 0.0f };
}

public class Plam
{
    public float x = 0.0f, y = 0.0f, z = 0.0f;
    public float roll = 0.0f, yaw = 0.0f, pitch = 0.0f;
}

public class main : MonoBehaviour
{
    public GameObject palam;
    public GameObject trapizum;
    public GameObject thumb_MCP;
    public GameObject thumb_IP;
    public GameObject fore_MCP;
    public GameObject fore_PIP;
    public GameObject fore_DIP;
    public GameObject middle_MCP;
    public GameObject middle_PIP;
    public GameObject middle_DIP;
    public GameObject ring_MCP;
    public GameObject ring_PIP;
    public GameObject ring_DIP;
    public GameObject little_MCP;
    public GameObject little_PIP;
    public GameObject little_DIP;

    public Finger forefinger;
    public Finger middlefinger;
    public Finger ringfinger;
    public Finger littlefinger;
    public Thumb thumb;
    public Plam plam;

    //读到的串口数据转换值
    public float[] accumulate = new float[30];

    void GetSignal()
    { }

    void GetAngle()
    {
        forefinger.DIP[0] = forefinger.PIP[0] * 2 / 3;
        middlefinger.DIP[0] = middlefinger.PIP[0] * 2 / 3;
        ringfinger.DIP[0] = ringfinger.PIP[0] * 2 / 3;
        littlefinger.DIP[0] = littlefinger.PIP[0] * 2 / 3;
        thumb.trapizum_LR[0] = thumb.MCP_LR[0] / 3;
        thumb.MCP_UD[0] = thumb.IP[0] / 2;
    }

    void GetTransform()
    {

        fore_MCP.transform.localEulerAngles = new Vector3(forefinger.MCP_UD[1], 0, 0);
    }

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        GetSignal();
        GetAngle();
        GetTransform();
    }
}

