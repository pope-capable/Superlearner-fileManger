'use strict';
import { DataTypes } from 'sequelize';
import db from '../../utils/db';
import {} from 'iyasunday';

const TABLE = 'folders',
  schema = {
    userId: {
      type: DataTypes.INTEGER(),
      allowNull: false,
      validate : {
        notNull : {msg : "Kindly suply userId"}
      }
    },

    name: {
      type: DataTypes.STRING(),
      allowNull: false
    },

    slug: {
      type: DataTypes.STRING(),
      allowNull: false
    }
  };

const Folder = db.define(TABLE, schema);
Folder.table = TABLE;

export default Folder;
