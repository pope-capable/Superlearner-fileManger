'use strict';
import { DataTypes } from 'sequelize';
import db from '../../utils/db';
import Folder from '../folder/model';

const TABLE = 'files',
  schema = {
    name: {
      type: DataTypes.STRING(),
      allowNull: false
    },

    description: {
      type: DataTypes.STRING(),
      allowNull: true
    },

    slug: {
      type: DataTypes.STRING(),
      allowNull: false
    },

    file: {
      type: DataTypes.STRING(),
      allowNull: false,
      get() {
        return process.env.FILE_URL+'/'+this.getDataValue('file')
      },
    }
  };

const File = db.define(TABLE, schema);
File.table = TABLE;

Folder.hasMany(File,{onDelete : 'cascade'});
File.belongsTo(Folder,{onDelete : 'cascade'})

export default File;
