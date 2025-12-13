using System.Collections.Generic;
using UnityEngine;


[System.Serializable]
public struct OHE_Elements
{
    public int position;
    public int count;

    public OHE_Elements(int p, int c)
    {
        position = p;
        count = c;
    }
}

public class OneHotEncoding
{
    List<OHE_Elements> elements;
    Dictionary<int, int> extraElements;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public OneHotEncoding(List<OHE_Elements> e)
    {
        elements = e;
        extraElements = new Dictionary<int, int>();
        for (int i = 0; i < elements.Count; i++)
        {
            int pos = elements[i].position;
            int c = elements[i].count;
            extraElements.Add(pos, c);
        }
    }

    /// <summary>
    /// Realiza la trasformación del OHE a los elementos seleccionados.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public float[] Transform(float[] input)
    {
        List<float> output = new List<float>();
        for (int i = 0; i < input.Length; i++)//recorrer input
        {
            //TODO implementar el OHE.
            if (extraElements.ContainsKey(i))//si la posicion esta en la lista para aplicar oneHotEncoding
            {
                for(int j = 0; j< extraElements.GetValueOrDefault(i); j++)//bucle del tamaño en el que se tiene que hacer oneHotEncoding
                {
                    if (input[i] == j)//si el valor de input es el mismo que la columna se añade 1
                    {
                        output.Add(1);
                    }
                    else//si no se añade 0
                    {
                        output.Add(0);
                    }
                }
            }
            else//si no hay que hacer oneHotEncoding se añade como esta
            {
                output.Add(input[i]);
            }
        }
        return output.ToArray();
    }
}
